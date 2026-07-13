import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
scene=S.make_scene(albedo='ramp')
p=mi.traverse(scene)
akey=[k for k in p.keys() if 'albedo.data' in k][0]
dirn=np.ones(np.array(p[akey]).size)
CFG=[('prb',        {'type':'prbvolpath','max_depth':32,'rr_depth':1032}),
     ('quad_defer', {**S.BASE}),
     ('quad_nodefer',{**S.BASE,'defer_probes':False}),
     ('quad_nomis', {**S.BASE,'use_probe_mis':False}),
     ('quad_pywalk',{**S.BASE,'real_interaction_cpp':False,
                     'real_interaction_fused':False}),
     ('lin_defer',  {**S.BASE,'linear_cost':True})]
for nm,d in CFG:
    ad,se,fd,fds=S.grad_param(scene,d,akey,dirn,fd_eps=5e-3,n=48)
    print(f'[{nm:12s}] ad={ad:+.4e} ± {se:.1e}   fd={fd:+.4e}  ratio={ad/fd:.3f}',flush=True)
print('DIAG ALBEDO DONE')
