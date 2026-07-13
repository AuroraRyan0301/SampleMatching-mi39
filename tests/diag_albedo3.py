import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
scene=S.make_scene(albedo='ramp')
p=mi.traverse(scene)
akey=[k for k in p.keys() if 'albedo.data' in k][0]
# fd reference from prior run: +3.797e+03
for src in (True,False):   # True=ratio, False=get_albedo
    for nm,d in [('defer  ',{**S.BASE}),('nodefer',{**S.BASE,'defer_probes':False})]:
        integ=mi.load_dict(d); integ._albedo_ratio=src
        dr.enable_grad(p[akey]); p.update()
        img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=1,seed_grad=2)
        dr.backward(dr.sum(img,axis=None))
        g=np.array(dr.grad(p[akey])).sum()
        dr.disable_grad(p[akey])
        print(f'[{"ratio " if src else "vcall "}{nm}] grad_sum={g:+.3e}  (fd ref +3.797e+03)',flush=True)
print('DBG DONE')
