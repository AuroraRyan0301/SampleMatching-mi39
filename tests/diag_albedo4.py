import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
CFG=[('defer_cpp_mis   ',{}),
     ('defer_cpp_nomis ',{'use_probe_mis':False}),
     ('defer_py_mis    ',{'real_interaction_fused':False,'real_interaction_cpp':False}),
     ('nodefer_cpp_mis ',{'defer_probes':False}),
     ('nodefer_py_nomis',{'defer_probes':False,'use_probe_mis':False,
                          'real_interaction_fused':False,'real_interaction_cpp':False})]
for nm,extra in CFG:
    scene=S.make_scene(albedo='ramp')   # fresh scene per config (stale-grad pitfall)
    p=mi.traverse(scene)
    akey=[k for k in p.keys() if 'albedo.data' in k][0]
    base=np.array(p[akey],copy=True)
    ad,se,fd,fds=S.grad_param(scene,{**S.BASE,**extra},akey,base.copy(),n=24)
    print(f'[{nm}] ad={ad:+.4e}±{se:.1e} fd={fd:+.4e} ratio={ad/fd:.3f}',flush=True)
print('DIAG ALBEDO4 DONE')
