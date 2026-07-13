"""
Timing decomposition for prbvolpath_sm slowness: same scene/recipe, 50 iters
each of prb / smlin(L1,2,4) / sm-quadratic(L4). Reports s/iter (excluding JIT
warmup) to separate 'probe walks' vs 'suffix' vs 'main path' costs.
"""
import time, sys, os
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb')
ROOT = '/path/to/SMmi39'
sys.path.insert(0, f'{ROOT}/pixel_batching_mi39')
from render_batch_mi39 import render_batch

DOWN, N, BATCH, SPPP, SPPG = 4, 50, 16384, 256, 8
RESX, RESY = 768//DOWN, 576//DOWN
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                     resx=RESX, resy=RESY, parallel=False)
params = mi.traverse(scene)
KS, KA = 'medium1.sigma_t.data', 'medium1.albedo.data'
import os
TRAINED = os.environ.get('TRAINED_SIGMA', '')
if TRAINED:
    sig0 = np.load(TRAINED).astype(np.float32)   # trained (dense) state
    print(f'using trained sigma grid: mean {sig0.mean():.4f} max {sig0.max():.3f}')
else:
    sig0 = np.full((64,)*3+(1,), 0.04/60, np.float32)
params[KS] = mi.TensorXf(sig0)
params[KA] = mi.TensorXf(np.full((64,)*3+(3,), 0.6, np.float32))
params.update()
def refs():
    out=[]
    for i in range(1,64):
        b=mi.Bitmap(f'{ROOT}/data/mi_ref/bunny-cloud/ref_{i:06d}.exr').convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.Float32,False)
        a=np.array(b); h,w=RESY*DOWN,RESX*DOWN
        out.append(a[:h,:w].reshape(RESY,DOWN,RESX,DOWN,-1).mean(axis=(1,3)))
    return np.stack(out)
R = refs()
rf = [mi.Float(np.ascontiguousarray(R[...,c]).ravel()) for c in range(3)]
sensors_train = dr.gather(mi.SensorPtr, scene.sensors_dr(), mi.UInt32(list(range(1,64))))

CONFIGS = [
  ('prb',        {'type':'prbvolpath','max_depth':32}),
  ('smlin-L1',   {'type':'prbvolpath_sm','max_depth':32,'probes_per_segment':1,'linear_cost':True}),
  ('smlin-L2',   {'type':'prbvolpath_sm','max_depth':32,'probes_per_segment':2,'linear_cost':True}),
  ('smlin-L4',   {'type':'prbvolpath_sm','max_depth':32,'probes_per_segment':4,'linear_cost':True}),
  ('smquad-L4',  {'type':'prbvolpath_sm','max_depth':32,'probes_per_segment':4}),
]
print(f'batch {BATCH}, spp {SPPP}/{SPPG}, film {RESX}x{RESY}, grid 64^3, 50 iters each')
for name, idict in CONFIGS:
    integ = mi.load_dict(idict)
    opt = mi.ad.Adam(lr=6e-3); opt[KS]=params[KS]; opt[KA]=params[KA]; params.update(opt)
    times=[]
    for it in range(N):
        t0=time.time()
        img, sidx, pix = render_batch(BATCH, scene, (RESX,RESY), params=params,
                                      integrator=integ, seed=1+2*it, spp=SPPP, spp_grad=SPPG,
                                      sensors=sensors_train, return_coords=True)
        px=dr.minimum(mi.UInt32(pix.x),RESX-1); py=dr.minimum(mi.UInt32(pix.y),RESY-1)
        lin=(sidx*RESY+py)*RESX+px
        rgb=mi.Vector3f(*[dr.gather(mi.Float,rf[c],lin) for c in range(3)])
        loss=dr.mean(dr.abs(img-mi.TensorXf(dr.ravel(rgb),shape=(1,BATCH,3))),axis=None)
        dr.backward(loss); opt.step()
        opt[KS]=dr.clip(opt[KS],0.0,250.0/60); opt[KA]=dr.clip(opt[KA],0.0,1.0)
        params.update(opt); dr.sync_thread()
        times.append(time.time()-t0)
    ts=np.array(times[5:])  # skip JIT warmup
    print(f'{name:11s}: {ts.mean():6.3f} s/iter (med {np.median(ts):6.3f}, warmup {times[0]:5.1f}s)', flush=True)
    # reset params for next method
    params[KS]=mi.TensorXf(sig0)
    params[KA]=mi.TensorXf(np.full((64,)*3+(3,),0.6,np.float32)); params.update()
