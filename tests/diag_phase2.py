import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
import importlib, stress_sm_lib
# make_scene in lib lacks phase_g; build scene via driver copy trick
sys.argv=['x','none']
def make_scene_hg(g):
    d_scene=S.make_scene()
    return d_scene  # placeholder not used
# construct directly:
import stress_sm_lib as L2
for kn,kv in [('prb_full',{}),('prb_no_nee',{'_kill_nee_phase':True}),
              ('prb_no_lo',{'_kill_phase_lo':True}),
              ('sm_full',{}),('sm_no_nee',{'_kill_nee_phase':True}),
              ('sm_no_lo',{'_kill_phase_lo':True})]:
    scene=None
    # rebuild scene with hg phase by patching interior via dict path in lib
    import mitsuba as mi2
    T=mi.ScalarTransform4f
    d={'type':'scene','integrator':{'type':'volpath'},
       'sensor':{'type':'perspective','fov':45,
                 'to_world':T().look_at([0,0,4.2],[0,0,0],[0,1,0]),
                 'film':{'type':'hdrfilm','width':64,'height':64,
                         'rfilter':{'type':'box'},'pixel_format':'rgb'},
                 'sampler':{'type':'independent','sample_count':8}},
       'light':{'type':'constant','radiance':1.0},
       'medium_shape':{'type':'cube','bsdf':{'type':'null'},
         'interior':{'type':'heterogeneous',
           'sigma_t':{'type':'gridvolume','grid':mi.VolumeGrid(S.sigma_grid()),
                      'to_world':T().translate([-1,-1,-1]).scale(2.0)},
           'albedo':{'type':'gridvolume','grid':mi.VolumeGrid(S.albedo_ramp()),
                     'to_world':T().translate([-1,-1,-1]).scale(2.0)},
           'scale':2.0,'majorant_resolution_factor':8,'majorant_factor':1.3,
           'phase':{'type':'hg','g':0.5},'sample_emitters':True}}}
    scene=mi.load_dict(d)
    p=mi.traverse(scene)
    gkey=[k for k in p.keys() if k.endswith('.g')][0]
    idict=dict(S.BASE) if not kn.startswith('prb') else {'type':'prbvolpath','max_depth':32,'rr_depth':1032}
    integ=mi.load_dict(idict)
    for a,v in kv.items(): setattr(integ,a,v)
    acc=[]
    for s_ in range(12):
        dr.enable_grad(p[gkey]); p.update()
        img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=s_,seed_grad=500+s_)
        dr.backward(dr.sum(img,axis=None))
        g=np.array(dr.grad(p[gkey])); dr.disable_grad(p[gkey])
        acc.append(float(np.where(np.isfinite(g),g,0).sum())*0.5)  # direction=g0=0.5
    print(f'[{kn:14s}] ad={np.mean(acc):+.4e}±{np.std(acc)/np.sqrt(12):.1e}  (fd ref -44.3)',flush=True)
print('PHASE DONE')
