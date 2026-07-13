import sys, os, numpy as np, drjit as dr, mitsuba as mi
variant=os.environ.get('MIVAR','llvm_ad_rgb')
mi.set_variant(variant); dr.set_thread_count(32)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
scene=S.make_scene(albedo='ramp')
p=mi.traverse(scene)
akey=[k for k in p.keys() if 'albedo.data' in k][0]
base=np.array(p[akey],copy=True)
integ=mi.load_dict({**S.BASE,'linear_cost':True}); integ._dbg=True
for it in range(2):
    dr.enable_grad(p[akey]); p.update()
    img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=500+it)
    dr.backward(dr.sum(img,axis=None))
    g=np.array(dr.grad(p[akey])); dr.disable_grad(p[akey])
    g=np.where(np.isfinite(g),g,0)
    print(f'[{variant} iter{it}] <g,base>={(g*base.ravel().reshape(g.shape)).sum():+.3e}',flush=True)
print('LINSFX DONE')
