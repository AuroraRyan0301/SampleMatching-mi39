"""FORWARD difference (avoids sigma>=0 clamp): [I(s+eps)-I(s)]/eps, eps sweep."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39-neeoff.xml',
    resx=768, resy=576, parallel=False,
    majorant_resolution_factor=8, majorant_factor=2.0,
    envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
    medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
    albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
base=np.array(p[K],copy=True)
vp=mi.load_dict({'type':'volpath','max_depth':64,'rr_depth':1064})
def rsum(seed):
    img=np.array(mi.render(scene,sensor=1,integrator=vp,spp=8,seed=seed))
    return np.where(np.isfinite(img),img,0).sum()
for EPS in (5e-4, 1e-3, 2e-3):
    vals=[]
    for it in range(16):
        seed,_=mi.sample_tea_32(13*it+1,0)
        p[K]=mi.TensorXf((base+EPS).astype(np.float32)); p.update(); hi=rsum(seed)
        p[K]=mi.TensorXf(base); p.update(); lo=rsum(seed)
        vals.append((hi-lo)/EPS)
    a=np.array(vals)
    print(f'[fwd eps={EPS:g}] FD<g,1>={a.mean():+.5e} +- {a.std(ddof=1)/np.sqrt(len(a)):.2e}',flush=True)
print('FD G1C DONE')
