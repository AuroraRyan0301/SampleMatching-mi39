import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
CFG=[('fused',   {}),
     ('naive',   {'real_interaction_fused':False,'real_interaction_cpp':True}),
     ('pywalk',  {'real_interaction_fused':False,'real_interaction_cpp':False}),
     ('nodefer', {'real_interaction_fused':False,'real_interaction_cpp':False,
                  'defer_probes':False})]
for b in ('sphere','torus'):
    scene=S.make_scene(boundary=b)
    p=mi.traverse(scene)
    key=[k for k in p.keys() if 'sigma_t.data' in k][0]
    base=np.array(p[key],copy=True)
    for nm,extra in CFG:
        d={**S.BASE,**extra}
        ad,se,fd,fds=S.grad_param(scene,d,key,base.copy(),n=48)
        print(f'[{b:6s} {nm:8s}] ad={ad:+.4e}±{se:.1e} fd={fd:+.4e} ratio={ad/fd:.3f}',flush=True)
print('DIAG BOUNDARY DONE')
