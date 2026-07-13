"""
Pixel batching for Mitsuba 3.9.0 (nanobind kernel) — a faithful port of the
unbiased-inverse-volume-rendering `batched.py::render_batch` to the modern API.

Ray-centric alternative to mi.render: sample `batch_size` random (sensor, pixel)
pairs across ALL sensors, render just those rays to a 1xN box-filter film, and
support a decorrelated PRB backward pass. Pure Python; no kernel changes.

Key mi3.0 -> mi3.9 API changes handled here:
  * integrator.sample(mode, scene, sampler, ray, δL, state_in, active, **kwargs)
    now returns a 4-tuple (L, valid, aovs, state)  [old: 3-tuple].
  * splatting via ADIntegrator._splat_to_block(...) + block.put(pos, [r,g,b,w]+aovs).
  * prbvolpath has no reparam; forward mode unsupported.
  * scene.sensors_dr() -> DynamicBuffer[SensorPtr]; vectorized SensorPtr vcall.
"""
from __future__ import annotations
import gc
from typing import Any
import drjit as dr
import mitsuba as mi
from mitsuba.ad.integrators.common import ADIntegrator


def sample_batch_pixels(batch_size, spp, spp_grad, sensors, film_size, seed):
    n_sensors = dr.width(sensors)
    batch_samplers = []
    for i, size in enumerate([batch_size, batch_size * spp, batch_size * spp_grad]):
        s = mi.load_dict({'type': 'independent'})
        s.seed(mi.sample_tea_32(seed, 17 * i + 5)[0], size)
        batch_samplers.append(s)
    sensor_idx = mi.UInt32(dr.minimum(n_sensors - 1,
                           mi.UInt32(n_sensors * batch_samplers[0].next_1d())))
    pixels = mi.Point2u(mi.Point2f(film_size) * batch_samplers[0].next_2d())
    return sensor_idx, pixels, batch_samplers


def sample_batch_rays(sampled_sensors, sampled_pixels, film_size, sampler, spp):
    batch_size = dr.width(sampled_pixels)
    repeat_idx = dr.arange(mi.UInt32, batch_size * spp) // spp
    sensors = dr.gather(type(sampled_sensors), sampled_sensors, repeat_idx)
    pos = dr.gather(type(sampled_pixels), sampled_pixels, repeat_idx)
    offset = sampler.next_2d()
    pos_f = mi.Vector2f(pos) + offset
    pos_unit = dr.rcp(mi.ScalarVector2f(film_size)) * pos_f
    wavelength_sample = sampler.next_1d() if mi.is_spectral else 0.0
    rays, ray_weights = sensors.sample_ray_differential(
        time=0.0, sample1=wavelength_sample, sample2=pos_unit, sample3=mi.Point2f(0.0))
    return rays, ray_weights, pos


def prepare_batch(rays, seed, spp, aovs, sampler=None):
    """1xN box-filter hdrfilm; box filter is the only valid choice (wavefront
    neighbourhood carries no image-space proximity)."""
    wavefront_size = dr.width(rays)
    assert wavefront_size % spp == 0
    film = mi.load_dict({'type': 'hdrfilm', 'width': wavefront_size // spp, 'height': 1,
                         'pixel_format': 'rgb', 'rfilter': {'type': 'box'}})
    sampler = mi.load_dict({'type': 'independent'}) if sampler is None else sampler.clone()
    sampler.set_sample_count(spp)
    sampler.set_samples_per_wavefront(spp)
    sampler.seed(seed, wavefront_size)
    film.prepare(aovs)
    return film, sampler, spp


def _develop_primal(integrator, film, sampler, scene, rays, spp):
    with dr.suspend_grad():
        L, valid, aovs, state = integrator.sample(
            dr.ADMode.Primal, scene, sampler, rays,
            δL=None, δaovs=None, state_in=None, active=mi.Bool(True))
        pos = mi.Point2f(0.5 + mi.Float(dr.arange(mi.UInt32, dr.width(rays)) // spp), 0.5)
        block = film.create_block()
        block.set_coalesce(block.coalesce() and spp >= 4)
        alpha = dr.select(valid, mi.Float(1), mi.Float(0))
        ADIntegrator._splat_to_block(block, film, pos, value=L, weight=mi.Float(1.0),
                                     alpha=alpha, aovs=aovs, wavelengths=rays.wavelengths)
        film.put_block(block)
        return film.develop(), state


def render_batch_backward(integrator, scene, grad_in, rays, seed, spp):
    aovs = integrator.aov_names()
    with dr.suspend_grad():
        film, sampler, spp = prepare_batch(rays, seed, spp, aovs)
        pos = mi.Point2f(0.5 + mi.Float(dr.arange(mi.UInt32, dr.width(rays)) // spp), 0.5)

        # Recover δL by differentiating the splat+develop of a dummy radiance.
        with dr.resume_grad():
            L = dr.full(mi.Spectrum, 1.0, dr.width(rays)); dr.enable_grad(L)
            block = film.create_block(); block.set_coalesce(block.coalesce() and spp >= 4)
            ADIntegrator._splat_to_block(block, film, pos, value=L, weight=mi.Float(1.0),
                                         alpha=mi.Float(1.0), aovs=[], wavelengths=rays.wavelengths)
            film.put_block(block)
            image = film.develop()
            dr.set_grad(image, grad_in); dr.enqueue(dr.ADMode.Backward, image)
            dr.traverse(dr.ADMode.Backward)
            δL = dr.grad(L)
        film.clear()

        # (1) primal replay to recover state, (2) adjoint deposit into scene params
        _, _, _, state = integrator.sample(dr.ADMode.Primal, scene, sampler.clone(), rays,
                                           δL=None, δaovs=None, state_in=None, active=mi.Bool(True))
        integrator.sample(dr.ADMode.Backward, scene, sampler, rays,
                          δL=δL, δaovs=None, state_in=state, active=mi.Bool(True))
        dr.eval()


class _BatchedRenderOp(dr.CustomOp):
    def eval(self, scene, sensors, sensors_idx, sampled_pixels, film_size,
             batch_samplers, _, params, integrator, seed, spp):
        # `_` is dict(params) (detached PyTree) so dr.custom detects attached
        # inputs; `params` is the SceneParameters used by the backward pass.
        self.scene, self.integrator = scene, integrator
        self.sampled_sensors = dr.gather(mi.SensorPtr, sensors, sensors_idx)
        self.sampled_pixels, self.film_size = sampled_pixels, film_size
        self.batch_samplers, self.params = batch_samplers, params
        self.seed, self.spp = seed, spp
        with dr.suspend_grad():
            rays, _, _ = sample_batch_rays(self.sampled_sensors, self.sampled_pixels,
                                           film_size, batch_samplers[1], spp[0])
            film, sampler, _ = prepare_batch(rays, seed[0], spp[0], integrator.aov_names())
            image, _ = _develop_primal(integrator, film, sampler, scene, rays, spp[0])
            return image

    def backward(self):
        rays, _, _ = sample_batch_rays(self.sampled_sensors, self.sampled_pixels,
                                       self.film_size, self.batch_samplers[2], self.spp[1])
        render_batch_backward(self.integrator, self.scene, self.grad_out(), rays,
                              seed=self.seed[1], spp=self.spp[1])

    def name(self):
        return "BatchedRenderOp"


def render_batch(batch_size, scene, film_size, params=None, integrator=None,
                 seed=0, seed_grad=0, spp=4, spp_grad=0, sensors=None,
                 return_coords=False):
    if integrator is None:
        integrator = scene.integrator()
    if spp_grad == 0:
        spp_grad = spp
    if seed_grad == 0:
        seed_grad = mi.sample_tea_32(seed, 1)[0]
    if sensors is None:
        sensors = scene.sensors_dr()
    film_size = mi.ScalarVector2u(film_size)
    sensors_idx, sampled_pixels, batch_samplers = sample_batch_pixels(
        batch_size, spp, spp_grad, sensors, film_size, seed)
    dict_params = dict(params) if params is not None else dict()
    img = dr.custom(_BatchedRenderOp, scene, sensors, sensors_idx, sampled_pixels,
                    film_size, batch_samplers, dict_params, params, integrator,
                    (seed, seed_grad), (spp, spp_grad))
    if return_coords:
        return img, dr.detach(sensors_idx), dr.detach(sampled_pixels)
    return img
