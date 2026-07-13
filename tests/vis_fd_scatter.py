import numpy as np, glob, struct
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT='/path/to/SMmi39'
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,c=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,c)
sig0=read_vol(f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params/00002000-medium1_sigma_t.vol')
IDX=sig0.shape[0]//2; sl=sig0[IDX,:,:,0].ravel()
mine=np.load(f'{ROOT}/grad_dump/mine_mean.npy').reshape(sig0.shape)[IDX,:,:,0].ravel()
ref =np.load(f'{ROOT}/grad_dump/ref_mean.npy').reshape(sig0.shape)[IDX,:,:,0].ravel()
fd=np.full(sl.size,np.nan)
for f in glob.glob(f'{ROOT}/grad_dump/fd_slice_part*.npy'):
    part=np.load(f); m=np.isfinite(part); fd[m]=part[m]
msk=np.isfinite(fd)&(sl>5e-3)
f_,a,b=fd[msk],mine[msk],ref[msk]
fig,axs=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
lim=np.percentile(np.abs(f_),98)
for ax,(d,t) in zip(axs,[(a,'mine (mi39 prbvolpath_sm)'),(b,'ref (old mi3 volpathfm)')]):
    s=(d*f_).sum()/(f_*f_).sum(); c=np.corrcoef(d,f_)[0,1]
    ax.plot(f_,d,'.',ms=3,alpha=0.4)
    xs=np.array([-lim,lim]); ax.plot(xs,xs,'k--',lw=1,label='y=x (unbiased)')
    ax.plot(xs,s*xs,'r-',lw=1,label=f'fit slope={s:.3f}')
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    ax.set_xlabel('FD per-voxel'); ax.set_title(f'{t}\ncorr={c:.2f} (FD noise attenuates both equally)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
axs[0].set_ylabel('adjoint mean per-voxel')
fig.suptitle(f'Occupied voxels on slice axis0@{IDX}, n={msk.sum()} (FD partial, spp64 CRN)')
fig.savefig(f'{ROOT}/inverse_results/fig10_fd_scatter.png',dpi=150,bbox_inches='tight')
print('saved fig10, n=',msk.sum())
