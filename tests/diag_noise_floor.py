"""Noise-floor diagnosis: is the 32.5 dB plateau reconstruction quality or
estimator/eval noise? Render the converged smlin state twice (indep seeds)
at several spp; render-vs-render PSNR bounds what render-vs-ref can measure."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                     resx=768, resy=576, parallel=False, majorant_resolution_factor=8)
p = mi.traverse(scene)
p['medium1.sigma_t.data'] = mi.TensorXf(np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_sigma.npy'))
p['medium1.albedo.data']  = mi.TensorXf(np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_albedo.npy'))
p.update()
integ = mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
bmp = mi.Bitmap(f'{ROOT}/data/mi_ref/bunny-cloud/ref_000000.exr').convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
ref = np.array(bmp)
def psnr(a,b): return -10*np.log10(max(float(np.mean((a-b)**2)),1e-12))
for spp in (256, 1024, 4096):
    ra = np.array(mi.render(scene, sensor=0, integrator=integ, spp=spp, seed=1))
    rb = np.array(mi.render(scene, sensor=0, integrator=integ, spp=spp, seed=2))
    avg = 0.5*(ra+rb)
    print(f'spp {spp:5d}: render-vs-render {psnr(ra,rb):6.2f} dB | render-vs-ref {psnr(ra,ref):6.2f} | avg2-vs-ref {psnr(avg,ref):6.2f}', flush=True)
