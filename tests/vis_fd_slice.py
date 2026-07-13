"""3-panel same-slice comparison: mine mean | ref mean | FD (ground truth)."""
import numpy as np, struct
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT='/path/to/SMmi39'
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,c=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,c)
sig0=read_vol(f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params/00002000-medium1_sigma_t.vol')
Z=sig0.shape[0]; IDX=Z//2
mine=np.load(f'{ROOT}/grad_dump/mine_mean.npy').reshape(sig0.shape)[IDX,:,:,0]
ref =np.load(f'{ROOT}/grad_dump/ref_mean.npy').reshape(sig0.shape)[IDX,:,:,0]
import glob
fd=np.full(mine.size,np.nan)
for f in glob.glob(f'{ROOT}/grad_dump/fd_slice_part*.npy'):
    part=np.load(f); m=np.isfinite(part); fd[m]=part[m]
done=int(np.isfinite(fd).sum()); print(f'FD voxels done: {done}/{fd.size}')
fd=fd.reshape(mine.shape)
# shared symmetric color scale from FD (the ground truth)
lim=np.percentile(np.abs(fd[np.isfinite(fd)&(fd!=0)]),99)
panels=[('mine (mi39 prbvolpath_sm)',mine),('ref (old mi3 volpathfm_linear_sd)',ref),('FD ground truth',fd)]
fig,axs=plt.subplots(1,3,figsize=(16,5.2))
for a,(t,d) in zip(axs,panels):
    im=a.imshow(d,cmap='RdBu_r',vmin=-lim,vmax=lim)
    a.set_title(t,fontsize=11); a.axis('off')
    # projection along sigma as a scalar summary
    m=np.isfinite(fd)
    proj=np.nansum(np.where(m,d,0.0)*sig0[IDX,:,:,0])
    a.text(0.5,-0.04,f'<slice,σ> = {proj:+.3e}',transform=a.transAxes,ha='center',va='top',fontsize=9)
fig.colorbar(im,ax=axs,shrink=0.7,aspect=30,label='dL/dσ_t (shared scale from FD)')
fig.suptitle(f'Per-voxel gradient, same slice (axis0 @ {IDX}/{Z}) — partial FD coverage; slice verdict: mine ~0.26x FD, ref ~1.2x FD',fontsize=12)
fig.savefig(f'{ROOT}/inverse_results/fig9_fd_vs_adjoint_slice.png',dpi=150,bbox_inches='tight')
print('saved fig9')
for t,d in panels:
    m=np.isfinite(fd)
    pr=np.nansum(np.where(m,d,0.0)*sig0[IDX,:,:,0]); pf=np.nansum(np.where(m,fd,0.0)*sig0[IDX,:,:,0])
    print(f'{t:38s}: <slice,sig>={pr:+.3e}  ratio-to-FD={pr/pf:+.2f}x')
