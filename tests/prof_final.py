"""OptiX performance + per-kernel breakdown: prb vs SM {quad, K4, linear},
best config (defer + fused C++ walk + majorant 1.3)."""
import time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=768,resy=576,parallel=False,
                   majorant_resolution_factor=8, majorant_factor=1.3,
                   use_bbox_fast_path='false',
                   envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
sig=np.array(p[K]).ravel().astype(np.float64)
B={'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4}
VAR=[('prb',{'type':'prbvolpath','max_depth':64,'rr_depth':1064}),
     ('quad',dict(B)),
     ('k4',{**B,'segment_reservoir':4}),
     ('lin',{**B,'linear_cost':True})]
dr.set_flag(dr.JitFlag.KernelHistory, True)
def one_iter(integ, seed):
    dr.enable_grad(p[K]); p.update()
    img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=seed,seed_grad=100+seed)
    dr.backward(dr.sum(img,axis=None))
    dr.eval(dr.grad(p[K])); dr.sync_thread(); dr.disable_grad(p[K])
for name,d in VAR:
    integ=mi.load_dict(d); ts=[]
    for it in range(3):
        t0=time.time(); one_iter(integ,it); ts.append(time.time()-t0)
    g=np.array(dr.grad(p[K]) if False else 0)
    dr.kernel_history_clear()
    t0=time.time(); one_iter(integ,7); wall=time.time()-t0
    hist=dr.kernel_history()
    kern=[h for h in hist if h.get('execution_time',0)>0.5]  # >0.5ms
    tot=sum(h.get('execution_time',0) for h in hist)
    print(f'==== {name}: steady={min(ts[1:]):.2f}s  profiled-iter={wall:.2f}s '
          f'(gpu kernel total {tot/1000:.2f}s) ====',flush=True)
    for i,h in enumerate(kern):
        print(f"  #{i:2d} size={h.get('size',0):>9d} ops={h.get('operation_count',0):>6d} "
              f"exec={h.get('execution_time',0):8.1f}ms ({100*h.get('execution_time',0)/max(tot,1e-9):4.1f}%)",flush=True)
print('PROF FINAL DONE')
