"""Gradient SNR in the dense core vs shell, at the converged formal state.
If core |mean|/std ~ 0 while shell is healthy -> deep-path gradient starvation."""
import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
sys.path.insert(0, f'{ROOT}/pixel_batching_mi39')
from render_batch_mi39 import render_batch

scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                     resx=768, resy=576, parallel=False, majorant_resolution_factor=8)
p = mi.traverse(scene)
K='medium1.sigma_t.data'
sig = np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_sigma.npy')
p[K] = mi.TensorXf(sig)
p['medium1.albedo.data'] = mi.TensorXf(np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_albedo.npy'))
p.update()
refs=[]
for i in range(1,64):
    b=mi.Bitmap(f'{ROOT}/data/mi_ref/bunny-cloud/ref_{i:06d}.exr').convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.Float32,False)
    refs.append(np.array(b))
refs_np=np.stack(refs)
rf=[mi.Float(np.ascontiguousarray(refs_np[...,c]).ravel()) for c in range(3)]
sensors=dr.gather(mi.SensorPtr, scene.sensors_dr(), mi.UInt32(list(range(1,64))))
integ = mi.load_dict({'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,
                      'probes_per_segment':4,'linear_cost':True})
RESX,RESY=768,576; N=16
grads=[]
for s in range(N):
    dr.enable_grad(p[K]); p.update()
    img,sidx,pix = render_batch(32768, scene,(RESX,RESY),params=p,integrator=integ,
                                seed=1+2*s,spp=256,spp_grad=16,sensors=sensors,return_coords=True)
    px=dr.minimum(mi.UInt32(pix.x),RESX-1); py=dr.minimum(mi.UInt32(pix.y),RESY-1)
    lin=(sidx*RESY+py)*RESX+px
    rgb=mi.Vector3f(*[dr.gather(mi.Float,rf[c],lin) for c in range(3)])
    loss=dr.mean(dr.abs(img-mi.TensorXf(dr.ravel(rgb),shape=(1,32768,3))),axis=None)
    dr.backward(loss)
    grads.append(np.array(dr.grad(p[K])).ravel()); dr.disable_grad(p[K])
    print(f'seed {s} done', flush=True)
G=np.stack(grads)                      # (N, voxels)
m, sd = G.mean(0), G.std(0)+1e-20
core = sig.ravel() > 0.35              # ~p99: the under-recovered dense region
mid  = (sig.ravel() > 0.05) & ~core
shell= (sig.ravel() > 0.005) & (sig.ravel() <= 0.05)
for name,mask in [('core >0.35',core),('mid 0.05-0.35',mid),('shell 0.005-0.05',shell)]:
    snr=np.abs(m[mask])/sd[mask]
    print(f'{name:18s}: n={mask.sum():7d}  median SNR {np.median(snr):.3f}  '
          f'mean grad {m[mask].mean():+.2e}  (neg frac {(m[mask]<0).mean():.0%})', flush=True)
