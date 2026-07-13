"""Kernel-level profile of prbvolpath_sm adjoint on bunny: per-iteration
kernel count, codegen vs execution time, cache hits, biggest kernels."""
import time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
dr.set_flag(dr.JitFlag.KernelHistory, True)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=768,resy=576,parallel=False,
                   majorant_resolution_factor=8, majorant_factor=2.0,
                   envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
for mname,d in [('sm',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4}),
                ('smlin',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True})]:
    integ=mi.load_dict(d)
    dr.kernel_history_clear()
    for it in range(4):
        t0=time.time()
        dr.enable_grad(p[K]); p.update()
        img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,
                      seed=it,seed_grad=100+it)
        dr.backward(dr.sum(img,axis=None))
        g=dr.grad(p[K]); dr.eval(g); dr.sync_thread()
        dr.disable_grad(p[K])
        wall=time.time()-t0
        hist=dr.kernel_history()
        kern=[h for h in hist if h.get('is_jit',h.get('backend',0)!=0)]
        cg=sum(h.get('codegen_time',0) for h in kern)
        ex=sum(h.get('execution_time',0) for h in kern)
        hits=sum(1 for h in kern if h.get('cache_hit',False))
        big=sorted(kern,key=lambda h:-h.get('execution_time',0))[:3]
        print(f'[{mname}] it{it}: wall={wall:6.1f}s kernels={len(kern):4d} cache_hit={hits:4d} '
              f'codegen={cg:8.1f}ms exec={ex:8.1f}ms',flush=True)
        for h in big:
            print(f'    top: size={h.get("size",0):9d} ops={h.get("operation_count",0):8d} '
                  f'codegen={h.get("codegen_time",0):7.1f}ms exec={h.get("execution_time",0):7.1f}ms '
                  f'hit={h.get("cache_hit",False)}',flush=True)
print('PROFILE DONE')
