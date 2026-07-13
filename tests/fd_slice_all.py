"""Per-voxel FD gradient on the axis-0 mid slice, ALL 65536 voxels (empty ones
included -- free-flight estimators are exactly the ones that miss gradients in
empty space, so those voxels are the most informative, not skippable).
Loss = sum(image) at 768x576, sensor 1 -- identical to the mean-dump protocol.
Correlated seeds (same seed both perturbations). Sharded via STRIDE/OFFSET."""
import os, time, numpy as np, struct, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
STRIDE=int(os.environ.get('FD_STRIDE',1)); OFF=int(os.environ.get('FD_OFFSET',0))
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,c=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,c)
sig0=read_vol(f'{CK}/00002000-medium1_sigma_t.vol')     # (Z,Y,X,1)
Z=sig0.shape[0]; IDX=Z//2
sl=sig0[IDX,:,:,0]; Y,X=sl.shape
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=768,resy=576,parallel=False,majorant_resolution_factor=8,
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
integ=mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
SPP=64; H_ABS=0.02; HREL=0.05
def render_sum(arr):
    p[K]=mi.TensorXf(arr.astype(np.float32)); p.update()
    return float(np.array(mi.render(scene,sensor=1,integrator=integ,spp=SPP,seed=7)).sum())
idx=np.arange(OFF, Y*X, STRIDE)
idx=idx[np.random.default_rng(0).permutation(len(idx))]  # partial run => uniform slice coverage
out=f'{ROOT}/grad_dump/fd_slice_part{OFF}.npy'
fd=np.load(out) if os.path.exists(out) else np.full(Y*X, np.nan)
idx=idx[~np.isfinite(fd[idx])]                            # resume: skip already-computed
print(f'part{OFF}: {len(idx)} voxels to go', flush=True)
t0=time.time()
for n,i in enumerate(idx):
    yy,xx=divmod(i,X); s=sl[yy,xx]
    h=max(HREL*s, H_ABS)
    hi=s+h; lo=max(s-h,0.0)
    a=sig0.copy(); a[IDX,yy,xx,0]=hi; Lp=render_sum(a)
    a=sig0.copy(); a[IDX,yy,xx,0]=lo; Lm=render_sum(a)
    fd[i]=(Lp-Lm)/(hi-lo)
    if n%100==0:
        el=time.time()-t0
        print(f'part{OFF}: {n}/{len(idx)}  {el/(n+1):.2f}s/vox  eta {(len(idx)-n)*el/(n+1)/60:.0f}min',flush=True)
        np.save(out,fd)
np.save(out,fd); print(f'part{OFF} done {time.time()-t0:.0f}s')
