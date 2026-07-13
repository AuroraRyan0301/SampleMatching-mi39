"""Primal-only render_batch throughput: integrator variants x spp."""
import sys, time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0, '/path/to/SMmi39/pixel_batching_mi39')
from render_batch_mi39 import render_batch
ROOT='/path/to/SMmi39'
RESX,RESY,BATCH=768,576,32768
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
    resx=RESX,resy=RESY,parallel=False,
    majorant_resolution_factor=8,majorant_factor=2.0,use_bbox_fast_path='true')
p=mi.traverse(scene)
sig=np.load(f'{ROOT}/inverse_results/smlin_fp1_fixed_formal_ckpt_sigma.npy')
alb=np.load(f'{ROOT}/inverse_results/smlin_fp1_fixed_formal_ckpt_albedo.npy')
p['medium1.sigma_t.data']=mi.TensorXf(sig); p['medium1.albedo.data']=mi.TensorXf(alb); p.update()
TRAIN=list(range(1,64))
sensors=dr.gather(mi.SensorPtr, scene.sensors_dr(), mi.UInt32(np.array(TRAIN,np.uint32)))
INTS=[('smlin',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True}),
      ('prbvolpath',{'type':'prbvolpath','max_depth':64,'rr_depth':1064}),
      ('volpath',{'type':'volpath','max_depth':64,'rr_depth':1064})]
for nm,d in INTS:
    integ=mi.load_dict(d)
    for spp in (1024, 128):
        ts=[]
        for it in range(4):
            t0=time.time()
            img,_,_=render_batch(BATCH,scene,(RESX,RESY),params=None,
                integrator=integ,seed=11+it,spp=spp,spp_grad=0,
                sensors=sensors,return_coords=True)
            dr.eval(img); dr.sync_thread()
            ts.append(time.time()-t0)
        best=min(ts[1:])
        print(f'[primal {nm:11s} spp={spp:5d}] {best:6.3f} s  ({BATCH*spp/best/1e6:6.1f} Msamp/s)',flush=True)
print('PRIMAL TP DONE')
