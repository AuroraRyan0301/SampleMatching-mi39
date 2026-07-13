import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('llvm_ad_rgb'); dr.set_thread_count(32)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
# lin on LLVM: albedo grads. Decompose: full lin, lin w/o suffix (max_depth clamp
# makes suffix vanish? no) -> use include-indirect toggle via quad comparison and
# suffix-only via difference. Simply: 3 renders each of lin & quad, <g,base>.
for nm,extra in [('lin ',{'linear_cost':True}),('quad',{'linear_cost':False})]:
    scene=S.make_scene(albedo='ramp')
    p=mi.traverse(scene)
    akey=[k for k in p.keys() if 'albedo.data' in k][0]
    base=np.array(p[akey],copy=True)
    integ=mi.load_dict({**S.BASE,**extra})
    for it in range(3):
        dr.enable_grad(p[akey]); p.update()
        img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=500+it)
        dr.backward(dr.sum(img,axis=None))
        g=np.array(dr.grad(p[akey])); dr.disable_grad(p[akey])
        g=np.where(np.isfinite(g),g,0)
        print(f'[{nm} iter{it}] <g,base>={(g*base.ravel().reshape(g.shape)).sum():+.3e}',flush=True)
print('LINLLVM DONE')
