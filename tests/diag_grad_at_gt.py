"""Decisive bias-vs-starvation probe: gradient of the training loss AT THE GT
state. Unbiased estimator -> coherent core gradient ~ 0; a coherent nonzero
core gradient pointing away from GT = practical bias in the deep-path regime."""
import sys, struct, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
sys.path.insert(0, f'{ROOT}/pixel_batching_mi39')
from render_batch_mi39 import render_batch

# GT scene (RGB sigma grid, GT albedo, same transforms the refs were rendered with)
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-ref-mi39.xml',
                     resx=768, resy=576, parallel=False)
p = mi.traverse(scene)
K = [k for k in p.keys() if k.endswith('sigma_t.data')][0]
print('sigma key:', K, 'shape:', p[K].shape, flush=True)
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
gt = np.array(p[K])          # (z,y,x,3)
gt_gray = gt.mean(-1).ravel()
# streaming mean/M2 (Welford) — 16 full 640^3 grids OOM'd a 50G node
mean_acc=None; m2_acc=None; cnt=0
for s in range(N):
    dr.enable_grad(p[K]); p.update()
    img,sidx,pix = render_batch(32768, scene,(RESX,RESY),params=p,integrator=integ,
                                seed=1+2*s,spp=256,spp_grad=16,sensors=sensors,return_coords=True)
    px=dr.minimum(mi.UInt32(pix.x),RESX-1); py=dr.minimum(mi.UInt32(pix.y),RESY-1)
    lin=(sidx*RESY+py)*RESX+px
    rgb=mi.Vector3f(*[dr.gather(mi.Float,rf[c],lin) for c in range(3)])
    loss=dr.mean(dr.abs(img-mi.TensorXf(dr.ravel(rgb),shape=(1,32768,3))),axis=None)
    dr.backward(loss)
    g = np.array(dr.grad(p[K])).mean(-1).ravel().astype(np.float32)
    dr.disable_grad(p[K])
    cnt += 1
    if mean_acc is None:
        mean_acc = g.copy(); m2_acc = np.zeros_like(g)
    else:
        d = g - mean_acc; mean_acc += d / cnt; m2_acc += d * (g - mean_acc)
    del g
    print(f'seed {s} done', flush=True)
m = mean_acc; sd = np.sqrt(m2_acc / max(cnt - 1, 1)) + 1e-20
core = gt_gray > 0.7          # GT p99 region (the part my run under-recovered)
mid  = (gt_gray > 0.1) & ~core
for name,mask in [('GT core >0.7',core),('GT mid 0.1-0.7',mid)]:
    snr=np.abs(m[mask])/sd[mask]
    print(f'{name:16s}: n={mask.sum():8d}  median SNR {np.median(snr):.3f}  '
          f'mean grad {m[mask].mean():+.3e}  (neg frac {(m[mask]<0).mean():.0%})', flush=True)
# Coherence test: is there a systematic push on the core?
t = m[core].mean() / (m[core].std()/np.sqrt(core.sum()))
print(f'core-mean t-stat: {t:+.1f}  (|t|>3 = coherent push, sign + = decrease density)')
