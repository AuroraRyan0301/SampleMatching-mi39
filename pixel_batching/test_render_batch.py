"""
End-to-end landing verification of pixel batching on mi39:
run a real inverse optimization THROUGH the render_batch CustomOp (forward+backward
+ Adam), recovering sigma_t of a homogeneous medium from a target batch brightness.
Proves the full _BatchedRenderOp (not just raw primitives) drives an optimizer.
"""
import drjit as dr, mitsuba as mi
mi.set_variant('llvm_ad_rgb')
import numpy as np
from render_batch_mi39 import render_batch

def make_sensor(o, up=[0,1,0]):
    return {'type':'perspective','fov':45,
            'to_world':mi.ScalarTransform4f().look_at(o,[0,0,0],up),
            'film':{'type':'hdrfilm','width':32,'height':32,'rfilter':{'type':'box'}}}

def build(sigma_t):
    return mi.load_dict({'type':'scene','integrator':{'type':'prbvolpath','max_depth':16},
        'light':{'type':'constant','radiance':1.0},
        's0':make_sensor([3,0,0]), 's1':make_sensor([0,2,2],up=[0,0,1]), 's2':make_sensor([0,0,3]),
        'box':{'type':'cube','bsdf':{'type':'null'},
               'interior':{'type':'homogeneous','sigma_t':sigma_t,'albedo':0.7}}})

FILM=(32,32); BATCH=4096; SPP=4

# --- target brightness = mean batch radiance of the GT scene (sigma_t=2.0) ---
gt = build(2.0)
tgt = float(np.array(render_batch(BATCH, gt, FILM, seed=999, spp=SPP)).mean())
print(f"target mean batch radiance (sigma_t=2.0): {tgt:.4f}")

# --- optimize sigma_t starting from 0.5 ---
scene = build(0.5)
params = mi.traverse(scene)
key = [k for k in params.keys() if k.endswith('sigma_t.value.value')][0]
print("optimizing:", key, "| init:", np.array(params[key]).ravel())

opt = mi.ad.Adam(lr=0.1)
opt[key] = params[key]
params.update(opt)

print("\n it |   sigma_t  |    loss")
for it in range(30):
    params.update(opt)
    img = render_batch(BATCH, scene, FILM, params=params, integrator=scene.integrator(),
                       seed=it, spp=SPP)
    loss = dr.sqr(dr.mean(img, axis=None) - tgt)
    dr.backward(loss)
    opt.step()
    opt[key] = dr.clip(opt[key], 0.01, 10.0)
    if it % 5 == 0 or it == 29:
        print(f" {it:2d} | {float(np.array(opt[key]).ravel()[0]):8.4f}   | {float(np.array(loss).ravel()[0]):.3e}")

final = float(np.array(opt[key]).ravel()[0])
print(f"\nrecovered sigma_t = {final:.3f}  (target 2.0)")
print("PIXEL-BATCH INVERSE OPTIMIZATION THROUGH render_batch CustomOp WORKS"
      if abs(final-2.0) < 0.4 else "converged but off target (check)")
