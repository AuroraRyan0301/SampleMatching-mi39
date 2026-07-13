"""Pure-volume fast-path benchmark: prb/sm/smlin adjoint s/iter with the
analytic-AABB fast path ON, plus correctness spot-check (fastpath on vs off:
primal sum and one-iteration <g,sigma> must agree closely)."""
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
# --- correctness: primal + 1-iter grad, fastpath off vs on
res={}
for fp in (False,True):
    scene=load(fp); p=mi.traverse(scene); K='medium1.sigma_t.data'
    sig=np.array(p[K]).ravel().astype(np.float64)
    integ=mi.load_dict({'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True})
    img=mi.render(scene,sensor=1,integrator=integ,spp=32,seed=7)
    dr.enable_grad(p[K]); p.update()
    im2=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=3,seed_grad=103)
    dr.backward(dr.sum(im2,axis=None))
    g=np.array(dr.grad(p[K])).ravel().astype(np.float64); dr.disable_grad(p[K])
    res[fp]=(float(np.array(img).sum()), float((g*sig).sum()))
    print(f'[check fp={fp}] primal_sum={res[fp][0]:.6e}  1-iter <g,sig>={res[fp][1]:+.5e}',flush=True)
print(f'[check] primal rel diff = {abs(res[True][0]-res[False][0])/abs(res[False][0]):.2e}',flush=True)
# --- speed with fast path ON
scene=load(True); p=mi.traverse(scene); K='medium1.sigma_t.data'
for name,d in [('prb',{'type':'prbvolpath','max_depth':64,'rr_depth':1064}),
               ('smlin',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True}),
               ('sm',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4})]:
    integ=mi.load_dict(d); ts=[]
    for it in range(4):
        t0=time.time()
        dr.enable_grad(p[K]); p.update()
        img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=100+it)
        dr.backward(dr.sum(img,axis=None))
        dr.eval(dr.grad(p[K])); dr.sync_thread(); dr.disable_grad(p[K])
        ts.append(time.time()-t0)
    print(f'[fastpath {name}] steady={min(ts[1:]):6.2f}s (it0 {ts[0]:.1f}s)   [old-mi3 targets: lin 1.7 / quad 2.8]',flush=True)
print('BENCH DONE')
