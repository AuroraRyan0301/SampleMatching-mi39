"""(1) FD gradcheck for ALBEDO gradients (never verified before!).
(2) Error-map decomposition of the formal smlin result vs reference."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'

# ---------- (1) albedo-direction FD gradcheck ----------
rng = np.random.default_rng(0)
GRID = rng.uniform(0.4, 1.6, size=(8,8,8,1)).astype(np.float32)
ALB  = rng.uniform(0.3, 0.9, size=(8,8,8,3)).astype(np.float32)
def build(idict, alb_scale=1.0):
    return mi.load_dict({'type':'scene','integrator':idict,
        'light':{'type':'constant','radiance':1.0},
        'sensor':{'type':'perspective','fov':45,
            'to_world':mi.ScalarTransform4f().look_at([0,0,4],[0,0,0],[0,1,0]),
            'film':{'type':'hdrfilm','width':24,'height':24,'rfilter':{'type':'box'}},
            'sampler':{'type':'independent','sample_count':8}},
        'box':{'type':'cube','bsdf':{'type':'null'},
            'interior':{'type':'heterogeneous',
                'sigma_t':{'type':'gridvolume','grid':mi.VolumeGrid(GRID),
                           'to_world':mi.ScalarTransform4f().translate([-1,-1,-1]).scale(2.0)},
                'albedo':{'type':'gridvolume','grid':mi.VolumeGrid(np.clip(ALB*alb_scale,0,0.999).astype(np.float32)),
                           'to_world':mi.ScalarTransform4f().translate([-1,-1,-1]).scale(2.0)},
                'scale':2.0}}})
H=0.03
def primal_loss(a_s, seed):
    sc=build({'type':'prbvolpath','max_depth':16,'rr_depth':1064}, a_s)
    return float(np.array(mi.render(sc,spp=256,seed=seed)).mean())
fd=np.mean([(primal_loss(1+H,100+s)-primal_loss(1-H,100+s))/(2*H) for s in range(4)])
def dirderiv(idict,n=8):
    sc=build(idict); p=mi.traverse(sc)
    key=[k for k in p.keys() if k.endswith('albedo.data')][0]
    vals=[]
    for s in range(n):
        dr.enable_grad(p[key]); p.update()
        img=mi.render(sc,p,spp=64,seed=10+s,seed_grad=1000+s)
        dr.backward(dr.mean(img,axis=None))
        g=np.array(dr.grad(p[key])).ravel(); dr.disable_grad(p[key])
        assert np.isfinite(g).all()
        vals.append(float((g*ALB.ravel()).sum()))
    return np.array(vals)
print('== (1) ALBEDO-direction FD gradcheck ==')
print(f'   FD : {fd:+.5e}')
for nm,idict in [('prb ',{'type':'prbvolpath','max_depth':16,'rr_depth':1064}),
                 ('sm  ',{'type':'prbvolpath_sm','max_depth':16,'rr_depth':1064,'probes_per_segment':4}),
                 ('smNM',{'type':'prbvolpath_sm','max_depth':16,'rr_depth':1064,'probes_per_segment':4,'use_probe_mis':False})]:
    v=dirderiv(idict)
    print(f'   {nm}: {v.mean():+.5e} (std {v.std():.1e})  err vs FD {abs(v.mean()-fd)/abs(fd):.1%}', flush=True)

# ---------- (2) error decomposition of the formal result ----------
print('== (2) formal smlin error decomposition (view 0) ==')
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                     resx=768, resy=576, parallel=False, majorant_resolution_factor=8)
p = mi.traverse(scene)
p['medium1.sigma_t.data'] = mi.TensorXf(np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_sigma.npy'))
p['medium1.albedo.data']  = mi.TensorXf(np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_albedo.npy'))
p.update()
integ = mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
img = np.array(mi.render(scene, sensor=0, integrator=integ, spp=1024, seed=7))
bmp = mi.Bitmap(f'{ROOT}/data/mi_ref/bunny-cloud/ref_000000.exr').convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.Float32,False)
ref = np.array(bmp)
err = ((img-ref)**2).mean(-1)
lum = ref.mean(-1)
cloud = lum > np.percentile(lum,60)   # bright cloud-ish region proxy
q = np.percentile(err,[50,90,99,99.9])
print(f'   MSE total {err.mean():.2e} -> PSNR {-10*np.log10(err.mean()):.2f}')
print(f'   err percentiles 50/90/99/99.9: {q[0]:.1e}/{q[1]:.1e}/{q[2]:.1e}/{q[3]:.1e}')
print(f'   top-1% pixels carry {np.sort(err.ravel())[-len(err.ravel())//100:].sum()/err.sum():.1%} of total error')
print(f'   bright-region MSE {err[cloud].mean():.2e} vs dark-region {err[~cloud].mean():.2e}')
mi.Bitmap(np.clip(np.sqrt(err/err.max()),0,1)[...,None].repeat(3,-1).astype(np.float32)).write(f'{ROOT}/inverse_results/formal_smlin_errmap.exr')
# sigma statistics vs GT (channel-mean, downsampled to 256)
import struct
with open(f'{ROOT}/data/scenes/bunny-cloud/bunny_cloud_512_rgb_unitbbox.vol','rb') as fh:
    h=fh.read(48); x,y,z,ch=struct.unpack('<4i',h[8:24]); gt=np.fromfile(fh,dtype='<f4').reshape(z,y,x,ch)
gt_gray = gt.mean(-1)
# crude downsample 640x640x512 -> compare distributions only
mine = np.load(f'{ROOT}/inverse_results/smlin_formal_ckpt_sigma.npy').ravel()
print(f'   sigma stats  mine: mean {mine.mean():.4f} p99 {np.percentile(mine,99):.3f} max {mine.max():.2f}')
print(f'   sigma stats  GT  : mean {gt_gray.mean():.4f} p99 {np.percentile(gt_gray,99):.3f} max {gt_gray.max():.2f}')
