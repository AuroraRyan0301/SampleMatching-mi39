"""Deferred-probe (two-stage adjoint) benchmark + correctness.
Grid: defer {off,on} x fastpath {off,on} x {smlin, sm}. N=64-seed mean
<g,sigma> must agree between defer off/on (different seeds -> ~1% tol)."""
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
BASE={'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4}
for fp in (True,False):
    scene=load(fp); p=mi.traverse(scene); K='medium1.sigma_t.data'
    sig=np.array(p[K]).ravel().astype(np.float64)
    for name,extra in [('smlin',{'linear_cost':True}),('sm',{})]:
        for defer in (False,True):
            integ=mi.load_dict({**BASE,**extra,'defer_probes':defer})
            ts=[]; acc=None; N=16
            for it in range(N):
                t0=time.time()
                dr.enable_grad(p[K]); p.update()
                img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,
                              seed=it,seed_grad=100+it)
                dr.backward(dr.sum(img,axis=None))
                gg=dr.grad(p[K]); dr.eval(gg); dr.sync_thread()
                g=np.array(gg).ravel().astype(np.float64)
                acc=g if acc is None else acc+g
                dr.disable_grad(p[K]); ts.append(time.time()-t0)
            gs=float(((acc/N)*sig).sum())
            print(f'[fp={int(fp)} {name} defer={int(defer)}] steady={min(ts[1:]):6.2f}s  '
                  f'N16<g,sig>={gs:+.5e}',flush=True)
print('DEFER BENCH DONE')
