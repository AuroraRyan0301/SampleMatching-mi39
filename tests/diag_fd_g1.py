"""FD referee for <dI_sum/dsigma_data, 1>: central difference along all-ones
direction in RAW data units, volpath primal, correlated seeds."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
for xml in ('bunny-cloud-mi39.xml','bunny-cloud-mi39-neeoff.xml'):
    scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/{xml}',
        resx=768, resy=576, parallel=False,
        majorant_resolution_factor=8, majorant_factor=2.0,
        envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
        medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
        albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
    p=mi.traverse(scene); K='medium1.sigma_t.data'
    base=np.array(p[K],copy=True); EPS=2e-3
    vp=mi.load_dict({'type':'volpath','max_depth':64,'rr_depth':1064})
    vals=[]
    for it in range(24):
        seed,_=mi.sample_tea_32(7*it+3,0)
        fs=[]
        for s in (+1.,-1.):
            p[K]=mi.TensorXf((base+s*EPS).astype(np.float32)); p.update()
            img=np.array(mi.render(scene,sensor=1,integrator=vp,spp=8,seed=seed))
            fs.append(np.where(np.isfinite(img),img,0).sum())
        vals.append((fs[0]-fs[1])/(2*EPS))
        if it%6==5:
            a=np.array(vals); print(f'[{xml}] it{it} FD<g,1>={a.mean():+.5e} +- {a.std(ddof=1)/np.sqrt(len(a)):.2e}',flush=True)
    p[K]=mi.TensorXf(base); p.update()
print('FD G1 DONE')
