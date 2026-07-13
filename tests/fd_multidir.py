"""Multi-direction FD arbiter: for several directions d_k, compute the true
directional derivative FD_k = <dL/dsigma, d_k> by central differences, and the
adjoint projections <mean_mine, d_k>, <mean_ref, d_k>. loss = sum(image)."""
import numpy as np, struct, drjit as dr, mitsuba as mi
from scipy.ndimage import gaussian_filter
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,c=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,c)
sig0=read_vol(f'{CK}/00002000-medium1_sigma_t.vol')          # (256,256,256,1)
mine=np.load(f'{ROOT}/grad_dump/mine_mean.npy').reshape(sig0.shape)
ref =np.load(f'{ROOT}/grad_dump/ref_mean.npy').reshape(sig0.shape)
s=sig0.squeeze(); rng=np.random.default_rng(0)
# directions (float32, only perturb non-empty voxels; keep sigma>=0 under +-h)
mask=(s>1e-4).astype(np.float32)
dirs={}
dirs['sigma']=s*mask
dirs['core']=((s>0.35)*s).astype(np.float32)
dirs['mid'] =(((s>0.05)&(s<=0.35))*s).astype(np.float32)
dirs['shell']=(((s>0.005)&(s<=0.05))*s).astype(np.float32)
for r in range(3):
    g=gaussian_filter(rng.standard_normal(s.shape).astype(np.float32),sigma=4)
    dirs[f'rand{r}']=(g*s).astype(np.float32)   # scale by sigma -> lives where medium is
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=384,resy=288,parallel=False,majorant_resolution_factor=8,
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'; integ=mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
SPP=4096
def loss(arr):
    p[K]=mi.TensorXf(arr.astype(np.float32)); p.update()
    return float(np.array(mi.render(scene,sensor=1,integrator=integ,spp=SPP,seed=7)).sum())
rows=[]
for name,d in dirs.items():
    rms_s=np.sqrt((s**2).mean()); rms_d=np.sqrt((d**2).mean())+1e-30
    h=0.02*rms_s/rms_d                             # comparable perturbation size
    d4=d[...,None]
    fd=(loss(sig0+h*d4)-loss(sig0-h*d4))/(2*h)
    pm=float((mine.squeeze()*d).sum()); pr=float((ref.squeeze()*d).sum())
    rows.append((name,fd,pm,pr))
    print(f'{name:6s}: FD {fd:+.4e} | mine {pm:+.4e} ({pm/fd:+.2f}x) | ref {pr:+.4e} ({pr/fd:+.2f}x)',flush=True)
np.save(f'{ROOT}/grad_dump/fd_multidir.npy', np.array([(f,m,r) for _,f,m,r in rows]))
np.save(f'{ROOT}/grad_dump/fd_multidir_names.npy', np.array([n for n,_,_,_ in rows]))
