import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
def run(tag,defer,kv,kf):
    scene=S.make_scene(albedo='ramp')
    p=mi.traverse(scene)
    akey=[k for k in p.keys() if 'albedo.data' in k][0]
    d=dict(S.BASE);
    if not defer: d['defer_probes']=False
    integ=mi.load_dict(d); integ._dbg=True; integ._kill_vertex=kv; integ._kill_flush=kf
    dr.enable_grad(p[akey]); p.update()
    img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=1,seed_grad=2)
    dr.backward(dr.sum(img,axis=None))
    g=np.array(dr.grad(p[akey])); base=np.array(p[akey])
    print(f'[{tag}] <g,1>={g.sum():+.3e} <g,base>={(g*base).sum():+.3e}',flush=True)
run('defer_full     ',True ,False,False)
run('defer_novertex ',True ,True ,False)
run('defer_noflush  ',True ,False,True )
run('nodefer_full   ',False,False,False)
run('nodefer_novertx',False,True ,False)
print('DBG6 DONE')
