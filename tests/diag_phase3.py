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

integ=mi.load_dict(dict(S.BASE))
integ._dbg_phase=True
for nm in ('_dbgL','_dbgC','_dbgP','_dbgN'):
    setattr(integ,nm,dr.zeros(mi.Float,2))
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
dr.enable_grad(p[gkey]); p.update()
img=mi.render(scene,p,sensor=0,integrator=integ,spp=8,spp_grad=8,seed=1,seed_grad=2)
dr.backward(dr.sum(img,axis=None))
L=np.array(integ._dbgL); C=np.array(integ._dbgC); P=np.array(integ._dbgP); N=np.array(integ._dbgN)
for i,tag in ((0,'primal '),(1,'adjoint')):
    print(f'[{tag}] N={N[i]:.0f} <L>={L[i]/max(N[i],1):.4f} <cos>={C[i]/max(N[i],1):.4f} <phase>={P[i]/max(N[i],1):.4f}',flush=True)
print('PHASE3 DONE')
