"""Quantify albedo-gradient behavior at EXACT-ZERO albedo voxels:
<g,1> on albedo, forward FD, use_probe_mis on/off + prb control."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
import sys; sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
scene=S.make_scene()
p=mi.traverse(scene)
key=[k for k in p.keys() if 'albedo.data' in k][0]
a=np.array(p[key],copy=True)
a[:4]=0.0                                # carve exact-zero albedo half
p[key]=mi.TensorXf(a.astype(np.float32)); p.update()
ones=np.ones_like(a)
for nm,extra in [('quad_mis',{}),('quad_nomis',{'use_probe_mis':False}),
                 ('lin_nomis',{'linear_cost':True,'use_probe_mis':False})]:
    d={**S.BASE,**extra}
    ad,se,fd,fds=S.grad_param(scene,d,key,ones,fd_eps=2e-3,fd_forward=True)
    print(f'[alb0 {nm:10s}] ad={ad:+.4e}±{se:.1e} fd={fd:+.4e} ratio={ad/fd:.3f}',flush=True)
ad,se,fd,fds=S.grad_param(scene,{'type':'prbvolpath','max_depth':32,'rr_depth':1032},key,ones,fd_eps=2e-3,fd_forward=True)
print(f'[alb0 prb       ] ad={ad:+.4e}±{se:.1e} fd={fd:+.4e} ratio={ad/fd:.3f}',flush=True)
print('ALB0 DONE')
