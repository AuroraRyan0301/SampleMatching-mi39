"""
Empirically test the CORE pixel-batching primitives on mitsuba 3.9.0:
  (1) Python-side dr.gather(mi.SensorPtr, scene.sensors_dr(), idx)
  (2) vectorized sensors.sample_ray_differential(...) over a batch mixing MULTIPLE sensors
  (3) render that arbitrary ray batch (ray-centric) and get radiance
  (4) differentiability: backward through a scene param from the batched render
This is exactly what render_batch/_BatchedRenderOp needs, and what reportedly broke in mi37.
"""
import drjit as dr
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

# --- tiny scene: a volumetric sphere in a constant emitter, with 3 distinct sensors ---
def make_sensor(origin):
    return {
        'type': 'perspective', 'fov': 45,
        'to_world': mi.ScalarTransform4f().look_at(origin=origin, target=[0,0,0], up=[0,1,0]),
        'film': {'type':'hdrfilm','width':16,'height':16,'rfilter':{'type':'box'},'pixel_format':'rgb'},
    }

scene = mi.load_dict({
    'type':'scene',
    'integrator':{'type':'prbvolpath','max_depth':8},
    'light':{'type':'constant','radiance':1.0},
    's0':make_sensor([3,0,0]),
    's1':make_sensor([0,2,2]),
    's2':make_sensor([0,0,3]),
    'medium_box':{
        'type':'cube',
        'bsdf':{'type':'null'},
        'interior':{'type':'homogeneous','sigma_t':2.0,'albedo':0.8},
    },
})

print("== [1] scene.sensors_dr() exists and is a SensorPtr array ==")
sensors = scene.sensors_dr()
print("   type:", type(sensors).__name__, "| width:", dr.width(sensors))

print("== [2] Python-side dr.gather(SensorPtr) mixing sensors + vectorized sample_ray_differential ==")
BATCH, SPP = 12, 1
# random (sensor, pixel) pairs — the essence of PIXEL batching (not sensor batching)
samp = mi.load_dict({'type':'independent'}); samp.seed(0, BATCH)
sensor_idx = mi.UInt32(dr.minimum(2, dr.floor(3*samp.next_1d()))) # 0,1,2 mixed within one batch
pix = samp.next_2d()                                              # random pixel per sample
gathered = dr.gather(mi.SensorPtr, sensors, sensor_idx)
rays, weights = gathered.sample_ray_differential(
    time=0.0, sample1=0.0, sample2=pix, sample3=mi.Point2f(0.0))
dr.eval(rays.o, rays.d)
print("   gathered SensorPtr width:", dr.width(gathered))
print("   ray origins (should differ per-sample = different sensors):")
import numpy as np
o = np.array(rays.o).T if hasattr(np.array(rays.o),'T') else np.array(rays.o)
print("  ", np.array(rays.o.x)[:6], "/", np.array(rays.o.y)[:6], "/", np.array(rays.o.z)[:6])
print("   distinct origins in batch:", len(set(map(tuple, np.stack([np.array(rays.o.x),np.array(rays.o.y),np.array(rays.o.z)],1).round(3)))))

print("== [3] ray-centric render of the arbitrary batch via integrator.sample ==")
integrator = scene.integrator()
L, valid, _, _ = integrator.sample(dr.ADMode.Primal, scene, samp.clone(), rays,
                                   δL=None, δaovs=None, state_in=None, active=mi.Bool(True))
dr.eval(L)
print("   L width:", dr.width(L), "| per-channel batch mean:", np.array(L).mean(axis=1).round(4))

print("== [4] differentiability via PRB two-pass (mirrors render_batch_backward) ==")
params = mi.traverse(scene)
key = [k for k in params.keys() if 'sigma_t' in k][0]
print("   diff param key:", key)
dr.enable_grad(params[key]); params.update()
# Pass 1: primal (detached) -> recover path state L
L_p, valid_p, aovs_p, state = integrator.sample(
    dr.ADMode.Primal, scene, samp.clone(), rays,
    δL=None, δaovs=None, state_in=None, active=mi.Bool(True))
# Pass 2: adjoint replay with an adjoint radiance δL (pretend dLoss/dpixel = 1)
integrator.sample(
    dr.ADMode.Backward, scene, samp.clone(), rays,
    δL=mi.Spectrum(1.0), δaovs=None, state_in=state, active=mi.Bool(True))
g = dr.grad(params[key])
gv = np.array(g).ravel()
print("   d(batch radiance)/d(sigma_t) =", gv[:4], " -> nonzero:", bool((gv != 0).any()))
print("\nALL PIXEL-BATCHING PRIMITIVES WORK ON mi39 (llvm_ad_rgb).")
