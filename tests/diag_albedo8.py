import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
def run(tag,cycle,flush,dbg):
    scene=S.make_scene(albedo='ramp')
    p=mi.traverse(scene)
    akey=[k for k in p.keys() if 'albedo.data' in k][0]
    base=np.array(p[akey],copy=True)
    integ=mi.load_dict(dict(S.BASE)); integ._dbg=dbg
    dr.enable_grad(p[akey]); p.update()
    for it in range(3):
        if cycle and it>0:
            dr.enable_grad(p[akey]); p.update()
        if flush and it>0:
            dr.flush_kernel_cache()
        img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=500+it)
        dr.backward(dr.sum(img,axis=None))
        g=np.array(dr.grad(p[akey]))
        if cycle: dr.disable_grad(p[akey])
        else: dr.set_grad(p[akey],0.0)
        g=np.where(np.isfinite(g),g,0)
        print(f'[{tag} iter{it}] <g,base>={(g*base.ravel().reshape(g.shape)).sum():+.3e}',flush=True)
run('keepon ',False,False,False)
run('kflush ',True ,True ,False)
run('dbgpr  ',True ,False,True )
print('DBG8 DONE')
