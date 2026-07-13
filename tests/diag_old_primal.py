"""Same uniform-light unified-majorant protocol, REFERENCE (old stack, LLVM)."""
import sys, numpy as np
sys.path.insert(0, '/path/to/PostTracking-opensource/python')
import drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb')
import drjit as _d; _d.set_thread_count(8)
from integrators.volpathfm_sd import VolpathFMSDIntegrator
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
SC='/path/to/PostTracking-opensource/data/scenes/bunny-cloud'
scene=mi.load_file(f'{SC}/bunny-cloud.xml', resx=768, resy=576,
                   majorant_resolution_factor=8, majorant_factor=2.0,
                   envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
sig=np.array(p[K]).ravel().astype(np.float64)
integ=mi.load_dict({'type':'volpathfm_sd','max_depth':64,'rr_depth':1064,
                    'use_drt':True,'use_drt_subsampling':False,'use_drt_mis':False,
                    'use_nee':True,'n_samples_transmittance':4})
integ.use_nee=True

def rsum(seed):
    img=np.array(mi.render(scene,params=p,sensor=1,integrator=integ,spp=8,seed=seed))
    return np.where(np.isfinite(img),img,0).sum(), int((~np.isfinite(img)).sum())
base=np.array(p[K]).astype(np.float32)
v,nb=rsum(7)
print(f'[old fm primal spp8 seed7] primal_sum={v:+.6e} nan_px={nb}',flush=True)
for EPS in (5e-4, 1e-3):
    vals=[]
    for it in range(12):
        seed,_=mi.sample_tea_32(13*it+1,0)
        p[K]=mi.TensorXf(base+EPS); p.update()
        hi,_=rsum(seed)
        p[K]=mi.TensorXf(base); p.update()
        lo,_=rsum(seed)
        vals.append((hi-lo)/EPS)
    a=np.array(vals)
    print(f'[old fwd eps={EPS:g}] FD<g,1>={a.mean():+.5e} +- {a.std(ddof=1)/np.sqrt(len(a)):.2e}',flush=True)
print('OLD PRIMAL DONE')
