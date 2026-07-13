"""K=4 striped-reservoir SM vs defer-quad vs defer-linear:
steady s/iter (fp1+fp0) + N=64 mean <g,sigma> vs quad anchor (-2.0705e5)."""
import time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def load(fp):
    return mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                        resx=768,resy=576,parallel=False,
                        majorant_resolution_factor=8, majorant_factor=1.3,
                        use_bbox_fast_path=('true' if fp else 'false'),
                        envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                        medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                        albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
B={'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4}
VAR=[('lin',{'linear_cost':True}),('lin_inner',{'linear_cost':True,'null_inner_loop':True}),('quad_inner',{'null_inner_loop':True})]
for fp in (True,False):
    scene=load(fp); p=mi.traverse(scene); K='medium1.sigma_t.data'
    sig=np.array(p[K]).ravel().astype(np.float64)
    for name,extra in VAR:
        integ=mi.load_dict({**B,**extra})
        ts=[]; acc=None; N=64 if fp else 16
        for it in range(N):
            t0=time.time()
            dr.enable_grad(p[K]); p.update()
            img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=100+it)
            dr.backward(dr.sum(img,axis=None))
            gg=dr.grad(p[K]); dr.eval(gg); dr.sync_thread()
            g=np.array(gg).ravel().astype(np.float64)
            acc=g if acc is None else acc+g
            dr.disable_grad(p[K]); ts.append(time.time()-t0)
        print(f'[fp={int(fp)} {name}] steady={min(ts[1:]):6.2f}s  N{N}<g,sig>={float(((acc/N)*sig).sum()):+.5e}',flush=True)
print('INNER BENCH DONE')
