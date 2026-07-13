"""Overnight stress campaign for prbvolpath_sm (fused C++ walk + deferred
probes, the PR configuration). Usage: stress_sm.py <group> [variant]

Groups: boundary | emitter | media | depthrr | lightsgrad | camera | speed
Each cell prints one structured line:
  CELL <group>/<name> <PASS|FAIL|WARN|INFO> key=value ...
and appends JSON to stress_results/<group>_<variant>.jsonl
"""
import sys, json, time, os
import numpy as np
import drjit as dr
import mitsuba as mi

GROUP = 'lib'
VARIANT='cuda_ad_rgb'

ROOT = '/path/to/SMmi39'
OUT = f'{ROOT}/stress_results/{GROUP}_{VARIANT}.jsonl'
T = mi.ScalarTransform4f

SM = {'quad': {}, 'lin': {'linear_cost': True}, 'k4': {'segment_reservoir': 4}}
BASE = {'type': 'prbvolpath_sm', 'max_depth': 32, 'rr_depth': 1032,
        'probes_per_segment': 4}

def emit(name, status, **kw):
    rec = {'cell': f'{GROUP}/{name}', 'status': status, **kw}
    print('CELL', rec['cell'], status,
          ' '.join(f'{k}={v}' for k, v in kw.items()), flush=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')

def sigma_grid():
    return np.load(f'{ROOT}/data/stress/sigma_blob.npy')

def albedo_ramp():
    return np.load(f'{ROOT}/data/stress/albedo_ramp.npy')

def make_scene(boundary='box', emitter='constant', albedo='ramp',
               homogeneous=False, supergrid=8, surface=False,
               res=64, spp=8):
    d = {'type': 'scene',
         'integrator': {'type': 'volpath'},
         'sensor': {'type': 'perspective', 'fov': 45,
                    'to_world': T().look_at([0, 0, 4.2], [0, 0, 0], [0, 1, 0]),
                    'film': {'type': 'hdrfilm', 'width': res, 'height': res,
                             'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
                    'sampler': {'type': 'independent', 'sample_count': spp}}}
    # ---- lighting
    if emitter == 'constant':
        d['light'] = {'type': 'constant', 'radiance': 1.0}
    elif emitter == 'envmap':
        d['light'] = {'type': 'envmap',
                      'filename': f'{ROOT}/data/scenes/common/uniform_white.exr',
                      'scale': 1.0}
    elif emitter == 'area_out':
        d['light'] = {'type': 'rectangle',
                      'to_world': T().translate([0, 2.5, 0]).rotate([1, 0, 0], 90).scale(1.2),
                      'emitter': {'type': 'area', 'radiance': 6.0}}
    elif emitter == 'area_in':
        d['light'] = {'type': 'rectangle',
                      'to_world': T().translate([0, 0, -0.5]).scale(0.25),
                      'emitter': {'type': 'area', 'radiance': 25.0}}
        d['fill'] = {'type': 'constant', 'radiance': 0.05}
    elif emitter == 'point':
        d['light'] = {'type': 'point', 'position': [2.5, 2.5, 2.5],
                      'intensity': 40.0}
        d['fill'] = {'type': 'constant', 'radiance': 0.02}
    # ---- medium
    if homogeneous:
        interior = {'type': 'homogeneous', 'sigma_t': 1.2, 'albedo': 0.8}
    else:
        interior = {'type': 'heterogeneous',
                    'sigma_t': {'type': 'gridvolume',
                                'grid': mi.VolumeGrid(sigma_grid()),
                                'to_world': T().translate([-1, -1, -1]).scale(2.0)},
                    'albedo': ({'type': 'gridvolume',
                                'grid': mi.VolumeGrid(albedo_ramp()),
                                'to_world': T().translate([-1, -1, -1]).scale(2.0)}
                               if albedo == 'ramp' else 0.8),
                    'scale': 2.0,
                    'majorant_resolution_factor': supergrid,
                    'majorant_factor': 1.3,
                    'sample_emitters': True}
    shape = {'bsdf': {'type': 'null'}, 'interior': interior}
    if boundary == 'box':
        shape.update({'type': 'cube'})
    elif boundary == 'sphere':
        shape.update({'type': 'sphere', 'radius': 1.15})
    elif boundary == 'torus':
        shape.update({'type': 'obj',
                      'filename': f'{ROOT}/data/stress/torus.obj',
                      'to_world': T().scale(1.4)})
    d['medium_shape'] = shape
    if surface:
        d['blocker'] = {'type': 'sphere', 'radius': 0.25,
                        'to_world': T().translate([0.4, 0.0, 0.0]),
                        'bsdf': {'type': 'diffuse',
                                 'reflectance': {'type': 'rgb',
                                                 'value': [0.7, 0.3, 0.2]}}}
    return mi.load_dict(d)

# --------------------------------------------------------------------------
def render_sum(scene, integ, spp, seed):
    img = mi.render(scene, sensor=0, integrator=integ, spp=spp, seed=seed)
    return float(np.array(img, copy=False).sum())

def primal_parity(scene, sm_dict, n=6, spp=256):
    ref = mi.load_dict({'type': 'volpath', 'max_depth': 32, 'rr_depth': 1032})
    it = mi.load_dict(sm_dict)
    a = np.array([render_sum(scene, ref, spp, 10 + s) for s in range(n)])
    b = np.array([render_sum(scene, it, spp, 10 + s) for s in range(n)])
    se = np.sqrt(a.var() / n + b.var() / n)
    z = abs(a.mean() - b.mean()) / max(se, 1e-9)
    rel = abs(a.mean() - b.mean()) / max(abs(a.mean()), 1e-9)
    return rel, z

def grad_param(scene, sm_dict, key, direction, n=48, spp=8,
               fd_eps=2e-2, fd_spp=2048, fd_int=None, fd_forward=False):
    """Adjoint <g, direction> (N-seed mean +- SE) vs central FD along
    `direction` with correlated seeds."""
    p = mi.traverse(scene)
    base = np.array(p[key], copy=True)
    integ = mi.load_dict(sm_dict)
    acc = []
    for s in range(n):
        dr.enable_grad(p[key]); p.update()
        img = mi.render(scene, p, sensor=0, integrator=integ,
                        spp=spp, spp_grad=spp, seed=s, seed_grad=500 + s)
        dr.backward(dr.sum(img, axis=None))
        g = np.array(dr.grad(p[key]), copy=False).ravel().astype(np.float64)
        dr.disable_grad(p[key])
        g = np.where(np.isfinite(g), g, 0)
        acc.append(float((g * direction.ravel()).sum()))
    acc = np.array(acc)
    ad, ad_se = acc.mean(), acc.std() / np.sqrt(n)
    # FD (primal only, unbiased referee)
    fd_int = fd_int or mi.load_dict({'type': 'volpath', 'max_depth': 32,
                                     'rr_depth': 1032})
    fds = []
    for s in (7, 8):
        hi = base + fd_eps * direction.reshape(base.shape)
        if fd_forward:
            lo, h2 = base, fd_eps
        else:
            lo = base - fd_eps * direction.reshape(base.shape)
            h2 = 2 * fd_eps
        p[key] = mi.TensorXf(hi.astype(np.float32)) if base.ndim > 1 else type(p[key])(hi)
        p.update()
        Lp = render_sum(scene, fd_int, fd_spp, s)
        p[key] = mi.TensorXf(lo.astype(np.float32)) if base.ndim > 1 else type(p[key])(lo)
        p.update()
        Lm = render_sum(scene, fd_int, fd_spp, s)
        fds.append((Lp - Lm) / h2)
    p[key] = mi.TensorXf(base.astype(np.float32)) if base.ndim > 1 else type(p[key])(base)
    p.update()
    fd = float(np.mean(fds))
    return ad, ad_se, fd, fds

def check_grad_sigma(scene, name, sm_dict):
    p = mi.traverse(scene)
    key = [k for k in p.keys() if 'sigma_t.data' in k][0]
    base = np.array(p[key], copy=True)
    dirn = base.copy()  # multiplicative direction: <g, sigma>
    ad, se, fd, fds = grad_param(scene, sm_dict, key, dirn)
    tol = 3 * se + 0.06 * abs(fd) + 0.5 * abs(fds[0] - fds[1])
    ok = abs(ad - fd) <= tol
    emit(name, 'PASS' if ok else 'FAIL', ad=f'{ad:.4e}', se=f'{se:.1e}',
         fd=f'{fd:.4e}', ratio=f'{ad/fd if fd else float("nan"):.3f}')
    return ok

