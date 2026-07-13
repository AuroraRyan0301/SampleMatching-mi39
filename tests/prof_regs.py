"""Extract per-kernel register/spill info from drjit's own JIT logs.
argv[1]=fp0|fp1. drjit's ptxas/driver JIT log lines carry 'registers' and
'spill' statistics when log level >= Debug."""
import sys, re, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
dr.set_log_level(dr.LogLevel.Debug)
fp = sys.argv[1]=='fp1'
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=768,resy=576,parallel=False,
                   majorant_resolution_factor=8, majorant_factor=2.0,
                   use_bbox_fast_path=('true' if fp else 'false'),
                   envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
integ=mi.load_dict({'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True})
dr.set_flag(dr.JitFlag.KernelHistory, True)
dr.enable_grad(p[K]); p.update()
img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=3,seed_grad=103)
dr.backward(dr.sum(img,axis=None))
dr.eval(dr.grad(p[K])); dr.sync_thread()
hist=dr.kernel_history()
big=sorted(hist,key=lambda h:-h.get('execution_time',0))[:3]
print(f'===== {sys.argv[1]} top kernels (all recorded fields) =====',flush=True)
for h in big:
    print({k:v for k,v in h.items() if not isinstance(v,(bytes,))},flush=True)
