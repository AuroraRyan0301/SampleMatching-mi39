"""Null-count speed scaling: majorant_factor 1.3 vs 2.0 (null share 23% vs 50%).
If steady time tracks null count, the C++ accept-until-real inner loop has
headroom; if flat, skip it (user's old-mi3 experience)."""
import time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def load(fp,mf):
    return mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                        resx=768,resy=576,parallel=False,
                        majorant_resolution_factor=8, majorant_factor=mf,
                        use_bbox_fast_path=('true' if fp else 'false'),
                        envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                        medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                        albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
for fp in (True,False):
    for mf in (2.0,1.3):
        scene=load(fp,mf); p=mi.traverse(scene); K='medium1.sigma_t.data'
        for name,d in [('prb',{'type':'prbvolpath','max_depth':64,'rr_depth':1064}),
                       ('smlin',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,
                                 'probes_per_segment':4,'linear_cost':True})]:
            integ=mi.load_dict(d); ts=[]
            for it in range(4):
                t0=time.time()
                dr.enable_grad(p[K]); p.update()
                img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=100+it)
                dr.backward(dr.sum(img,axis=None))
                dr.eval(dr.grad(p[K])); dr.sync_thread(); dr.disable_grad(p[K])
                ts.append(time.time()-t0)
            print(f'[fp={int(fp)} mf={mf} {name}] steady={min(ts[1:]):6.2f}s',flush=True)
print('MAJNULL DONE')
