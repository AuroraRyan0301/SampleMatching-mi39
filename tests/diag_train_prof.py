"""Per-phase timing of one formal-training step (smlin fp1, 256^3 state)."""
import os, sys, time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(16)
sys.path.insert(0, '/path/to/SMmi39/pixel_batching_mi39')
sys.path.insert(0, '/path/to/SMmi39/sm_integrator_tests')
from render_batch_mi39 import render_batch
ROOT='/path/to/SMmi39'
RESX, RESY, BATCH, SPPP, SPPG, LR = 768, 576, 32768, 1024, 16, 6e-3
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                     resx=RESX, resy=RESY, parallel=False,
                     majorant_resolution_factor=8, majorant_factor=2.0,
                     use_bbox_fast_path='true')
integrator = mi.load_dict({'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,
                           'probes_per_segment':4,'linear_cost':True})
params = mi.traverse(scene)
K_SIG='medium1.sigma_t.data'; K_ALB='medium1.albedo.data'
sig=np.load(f'{ROOT}/inverse_results/smlin_fp1_fixed_formal_ckpt_sigma.npy')
alb=np.load(f'{ROOT}/inverse_results/smlin_fp1_fixed_formal_ckpt_albedo.npy')
opt = mi.ad.Adam(lr=LR)
opt[K_SIG]=mi.TensorXf(sig); opt[K_ALB]=mi.TensorXf(alb)
params.update(opt)
print('grid', sig.shape, flush=True)
REF=f'{ROOT}/data/mi_ref/bunny-cloud'
TRAIN=list(range(1,64))
refs=np.stack([np.array(mi.Bitmap(f'{REF}/ref_{i:06d}.exr'))[:,:,:3] for i in TRAIN])
refs_flat=[mi.Float(np.ascontiguousarray(refs[...,c]).ravel()) for c in range(3)]
sensors_train=dr.gather(mi.SensorPtr, scene.sensors_dr(),
                        mi.UInt32(np.array(TRAIN,np.uint32)))
def S(): dr.sync_thread(); return time.time()
phases={'render':0.,'loss':0.,'backward':0.,'step':0.,'update':0.}
NW,NT=3,16
dr.set_flag(dr.JitFlag.KernelHistory, True)
hist_stats={'miss':0,'hit':0,'codegen':0.}
for it in range(NW+NT):
    timed = it>=NW
    if timed: dr.kernel_history()  # clear
    t=S()
    img,sidx,pix = render_batch(BATCH, scene, (RESX,RESY), params=params,
                                integrator=integrator, seed=1+2*it,
                                spp=SPPP, spp_grad=SPPG,
                                sensors=sensors_train, return_coords=True)
    t2=S(); phases['render']+= (t2-t) if timed else 0
    px=dr.minimum(mi.UInt32(pix.x),RESX-1); py=dr.minimum(mi.UInt32(pix.y),RESY-1)
    lin=(sidx*RESY+py)*RESX+px
    ref_rgb=mi.Vector3f(*[dr.gather(mi.Float,refs_flat[c],lin) for c in range(3)])
    ref_img=mi.TensorXf(dr.ravel(ref_rgb),shape=(1,BATCH,3))
    loss=dr.mean(dr.abs(img-ref_img),axis=None)
    t3=S(); phases['loss']+= (t3-t2) if timed else 0
    dr.backward(loss)
    t4=S(); phases['backward']+= (t4-t3) if timed else 0
    opt.step()
    opt[K_SIG]=dr.clip(opt[K_SIG],0.0,250.0); opt[K_ALB]=dr.clip(opt[K_ALB],0.0,1.0)
    t5=S(); phases['step']+= (t5-t4) if timed else 0
    params.update(opt)
    t6=S(); phases['update']+= (t6-t5) if timed else 0
    if timed:
        for k in dr.kernel_history():
            if k.get('type',None) is not None and 'JIT' not in str(k.get('type','')): continue
            if k.get('cache_hit', True): hist_stats['hit']+=1
            else:
                hist_stats['miss']+=1
                hist_stats['codegen']+=k.get('codegen_time',0.)
    if it==NW-1: print('warmup done',flush=True)
tot=sum(phases.values())/NT
print(f'--- per-iter breakdown over {NT} iters (total {tot:.3f} s):')
for k,v in phases.items(): print(f'  {k:9s} {v/NT:7.3f} s  ({100*v/NT/tot:4.1f}%)',flush=True)
print(f'--- kernels/iter: cache_hit {hist_stats["hit"]/NT:.1f}  MISS {hist_stats["miss"]/NT:.1f}  codegen {hist_stats["codegen"]/NT/1000.0:.3f} s/iter')
print('TRAIN PROF DONE')
