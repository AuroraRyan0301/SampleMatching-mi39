"""Round-1 optimization benchmark: NEE intersect w/ Minimal flags.
Measures smlin+sm adjoint s/iter with OptiX (fast path OFF = general-scene
proxy) and with fast path ON, plus 1-iter gradient consistency."""
import time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def load(fp):
    return mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                        resx=768,resy=576,parallel=False,
                        majorant_resolution_factor=8, majorant_factor=2.0,
                        use_bbox_fast_path=('true' if fp else 'false'),
                        envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                        medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                        albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
for fp,base in [(False,{'smlin':25.8,'sm':45.0}),(True,{'smlin':2.43,'sm':4.45})]:
    scene=load(fp); p=mi.traverse(scene); K='medium1.sigma_t.data'
    sig=np.array(p[K]).ravel().astype(np.float64)
    for name,d in [('smlin',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True}),
                   ('sm',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4})]:
        integ=mi.load_dict(d); ts=[]; g1=None
        for it in range(4):
            t0=time.time()
            dr.enable_grad(p[K]); p.update()
            img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=3,seed_grad=103)
            dr.backward(dr.sum(img,axis=None))
            g=dr.grad(p[K]); dr.eval(g); dr.sync_thread()
            if it==0: g1=float((np.array(g).ravel().astype(np.float64)*sig).sum())
            dr.disable_grad(p[K]); ts.append(time.time()-t0)
        print(f'[fp={int(fp)} {name}] steady={min(ts[1:]):6.2f}s  (was {base[name]:.2f}s)  1-iter<g,sig>={g1:+.5e}',flush=True)
print('OPT1 DONE')
