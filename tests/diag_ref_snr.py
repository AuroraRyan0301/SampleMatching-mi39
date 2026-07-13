"""Head-to-head gradient SNR: run the REFERENCE integrator (old mitsuba,
volpathfm_linear_sd) at MY converged 256^3 state, same loss/batch/spp as my
probe, and report core/mid/shell per-voxel SNR for direct comparison."""
import sys, numpy as np
sys.path.insert(0, '/path/to/PostTracking-opensource/python')
import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from integrators.volpathfm_linear_sd import VolpathFMLinearSDIntegrator  # registers plugin
from batched import render_batch

ROOT='/path/to/SMmi39'
SC='/path/to/PostTracking-opensource/data/scenes/bunny-cloud'
scene = mi.load_file(f'{SC}/bunny-cloud.xml', resx=768, resy=576,
                     majorant_resolution_factor=8)
p = mi.traverse(scene)
K='medium1.sigma_t.data'
sig = np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_sigma.npy')
p[K] = mi.TensorXf(sig)
p['medium1.albedo.data'] = mi.TensorXf(np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_albedo.npy'))
p.update()
integ = mi.load_dict({'type':'volpathfm_linear_sd','max_depth':64,'rr_depth':1064,
                      'use_drt':True,'use_drt_subsampling':True,'use_drt_mis':False,
                      'use_nee':True,'n_samples_transmittance':4})
refs=[]
for i in range(1,64):
    b=mi.Bitmap(f'{ROOT}/data/mi_ref/bunny-cloud/ref_{i:06d}.exr')
    b=b.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
    refs.append(np.array(b))
refs_np=np.stack(refs)
rf=[mi.Float(np.ascontiguousarray(refs_np[...,c]).ravel().astype(np.float32)) for c in range(3)]
sensors=dr.gather(mi.SensorPtr, scene.sensors_dr(), mi.UInt32(list(range(1,64))))
film_size=mi.ScalarVector2u(768,576)
RESX,RESY=768,576; N=16
mean_acc=None; m2_acc=None; cnt=0
for s in range(N):
    dr.enable_grad(p[K]); p.update()
    image, film, smp, sidx, pix = render_batch(
        32768, scene, sensors, film_size, params=p, integrator=integ,
        spp=256, spp_grad=16, seed=1+2*s, seed_grad=2+2*s)
    px=dr.minimum(mi.UInt32(pix[0] if isinstance(pix,tuple) else pix.x), RESX-1)
    py=dr.minimum(mi.UInt32(pix[1] if isinstance(pix,tuple) else pix.y), RESY-1)
    lin=(mi.UInt32(sidx)*RESY+py)*RESX+px
    r,g_,b_=[dr.gather(mi.Float, rf[c], lin) for c in range(3)]
    ref_t = mi.TensorXf(dr.ravel(mi.Vector3f(r,g_,b_)), shape=(1,32768,3))
    loss = dr.sum(dr.abs(image - ref_t)) / dr.width(image)
    dr.backward(loss)
    g = np.array(dr.grad(p[K])).ravel().astype(np.float32)
    dr.disable_grad(p[K])
    cnt+=1
    if mean_acc is None: mean_acc=g.copy(); m2_acc=np.zeros_like(g)
    else:
        d=g-mean_acc; mean_acc+=d/cnt; m2_acc+=d*(g-mean_acc)
    del g
    print(f'seed {s} done', flush=True)
m=mean_acc; sd=np.sqrt(m2_acc/max(cnt-1,1))+1e-20
sigr = sig.ravel()
for name,mask in [('core >0.35', sigr>0.35), ('mid 0.05-0.35',(sigr>0.05)&(sigr<=0.35)),
                  ('shell 0.005-0.05',(sigr>0.005)&(sigr<=0.05))]:
    snr=np.abs(m[mask])/sd[mask]
    print(f'REF {name:18s}: n={mask.sum():7d}  median SNR {np.median(snr):.3f}  '
          f'mean grad {m[mask].mean():+.2e}  (neg frac {(m[mask]<0).mean():.0%})', flush=True)
