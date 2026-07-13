"""FD refinement: eps sweep + NaN count + volpath vs prbvolpath primal, NEE-off scene."""
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
print('raw sigma stats: max',base.max(),'mean',base.mean(),flush=True)
def render_sum(integ,seed):
    img=np.array(mi.render(scene,sensor=1,integrator=integ,spp=8,seed=seed))
    bad=~np.isfinite(img)
    return np.where(bad,0,img).sum(), int(bad.sum())
vp=mi.load_dict({'type':'volpath','max_depth':64,'rr_depth':1064})
pb=mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
for EPS in (5e-4, 2e-3, 8e-3):
    for nm,integ in (('volpath',vp),('prb-primal',pb)):
        vals=[]; nbad=0
        for it in range(12):
            seed,_=mi.sample_tea_32(11*it+5,0)
            fs=[]
            for s in (+1.,-1.):
                p[K]=mi.TensorXf((base+s*EPS).astype(np.float32)); p.update()
                v,nb=render_sum(integ,seed); fs.append(v); nbad+=nb
            vals.append((fs[0]-fs[1])/(2*EPS))
        a=np.array(vals)
        print(f'[eps={EPS:g} {nm}] FD<g,1>={a.mean():+.5e} +- {a.std(ddof=1)/np.sqrt(len(a)):.2e} nan_px={nbad}',flush=True)
p[K]=mi.TensorXf(base); p.update()
print('FD G1B DONE')
