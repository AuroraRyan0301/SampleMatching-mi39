"""Same protocol, REFERENCE estimator (old mitsuba)."""
import sys, numpy as np
sys.path.insert(0, '/path/to/PostTracking-opensource/python')
import drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from integrators.volpathfm_linear_sd import VolpathFMLinearSDIntegrator
ROOT='/path/to/SMmi39'
CKPT='/path/to/SMmi39/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
SC='/path/to/PostTracking-opensource/data/scenes/bunny-cloud'
scene = mi.load_file(f'{SC}/bunny-cloud.xml', resx=384, resy=288,
                     majorant_resolution_factor=8,
                     medium_filename=f'{CKPT}/00002000-medium1_sigma_t.vol',
                     albedo_filename=f'{CKPT}/00002000-medium1_albedo.vol')
p = mi.traverse(scene); K='medium1.sigma_t.data'
integ = mi.load_dict({'type':'volpathfm_linear_sd','max_depth':64,'rr_depth':1064,
                      'use_drt':True,'use_drt_subsampling':True,'use_drt_mis':False,
                      'use_nee':True,'n_samples_transmittance':1})
integ.use_nee = True
N=512; SPP=8; CAM=1
mean=None; m2=None; mx=None; nonfinite=0
for it in range(N):
    seed,_ = mi.sample_tea_32(3*it+0, 0)
    seed_grad,_ = mi.sample_tea_32(3*it+1, 0)
    if seed==seed_grad: seed_grad += 100
    dr.enable_grad(p[K]); p.update()
    img = mi.render(scene, params=p, sensor=CAM, integrator=integ,
                    spp=SPP, spp_grad=SPP, seed=seed, seed_grad=seed_grad)
    dr.backward(dr.sum(img))
    g = np.array(dr.grad(p[K])).ravel().astype(np.float64)
    dr.disable_grad(p[K])
    bad=~np.isfinite(g); nonfinite+=int(bad.sum()); g=np.where(bad,0,g)
    if mean is None: mean=g.copy(); m2=np.zeros_like(g); mx=np.abs(g)
    else:
        c=it+1; d=g-mean; mean+=d/c; m2+=d*(g-mean); mx=np.maximum(mx,np.abs(g))
    if it%64==0 or it==N-1: print(f'iter {it} (nonfinite: {nonfinite})', flush=True)
std=np.sqrt(m2/(N-1))
np.save(f'{ROOT}/grad_dump/refN1_mean.npy', mean.astype(np.float32))
np.save(f'{ROOT}/grad_dump/refN1_std.npy', std.astype(np.float32))
np.save(f'{ROOT}/grad_dump/refN1_max.npy', mx.astype(np.float32))
print(f'TOTAL nonfinite: {nonfinite}')
