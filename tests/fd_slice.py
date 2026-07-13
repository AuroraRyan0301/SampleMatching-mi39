"""Per-voxel FD gradient field on ONE slice (axis-0 mid), so it can be shown
side-by-side with the mine/ref adjoint means on the same slice.
For each voxel v in the slice, FD_v = (L(sigma + h e_v) - L(sigma - h e_v))/(2h),
loss = sum(image). Batched: all voxels in the slice share one +/- render pair
by exploiting linearity is NOT valid (loss nonlinear), so we do it per-voxel via
a coordinate sweep — but batch the renders across many voxels using independent
perturbations is also invalid. => We perturb each voxel one at a time but only
for the slice's occupied voxels, using correlated (same-seed) renders."""
import numpy as np, struct, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,c=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,c)
sig0=read_vol(f'{CK}/00002000-medium1_sigma_t.vol')     # (Z,Y,X,1)
Z=sig0.shape[0]; AX=0; IDX=Z//2
sl_sigma = sig0[IDX,:,:,0]                               # (Y,X) slice values
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=384,resy=288,parallel=False,majorant_resolution_factor=8,
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'; integ=mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
SPP=256
def render_sum(arr):
    p[K]=mi.TensorXf(arr.astype(np.float32)); p.update()
    return float(np.array(mi.render(scene,sensor=1,integrator=integ,spp=SPP,seed=7)).sum())
Y,X = sl_sigma.shape
fd = np.zeros((Y,X), np.float64)
# only bother with occupied voxels of the slice (empty -> ~0 grad, skip to save time)
ys,xs = np.where(sl_sigma > 5e-3)
print(f'slice {IDX}: {len(ys)} occupied voxels of {Y*X}', flush=True)
Hrel=0.05
for n,(yy,xx) in enumerate(zip(ys,xs)):
    h = Hrel*max(sl_sigma[yy,xx], 1e-3)
    a=sig0.copy(); a[IDX,yy,xx,0]+=h; Lp=render_sum(a)
    a=sig0.copy(); a[IDX,yy,xx,0]-=h; Lm=render_sum(a)
    fd[yy,xx]=(Lp-Lm)/(2*h)
    if n%500==0:
        print(f'{n}/{len(ys)}',flush=True)
        np.save(f'{ROOT}/grad_dump/fd_slice.npy', fd)
np.save(f'{ROOT}/grad_dump/fd_slice.npy', fd)
np.save(f'{ROOT}/grad_dump/fd_slice_meta.npy', np.array([AX,IDX]))
print('done')
