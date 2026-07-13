import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
scene=S.make_scene(albedo='ramp')
p=mi.traverse(scene)
akey=[k for k in p.keys() if 'albedo.data' in k][0]
for nm,d in [('defer_on',{**S.BASE}),('defer_off',{**S.BASE,'defer_probes':False})]:
    integ=mi.load_dict(d)
    dr.enable_grad(p[akey]); p.update()
    print(f'--- {nm}',flush=True)
    img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=1,seed_grad=2)
    dr.backward(dr.sum(img,axis=None))
    g=np.array(dr.grad(p[akey])).sum()
    dr.disable_grad(p[akey])
    print(f'[{nm}] grad_sum={g:+.3e}',flush=True)
print('DBG DONE')
