"""Same-slice comparison: original DRT (volpathsimple) quad vs linear, next to
the fm-fork quad and linear. Uniform-light protocol, slice axis0@128, CubicL."""
import numpy as np, struct
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
from cubicl import cubicL
ROOT='/path/to/SMmi39'
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,c=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,c)
sig0=read_vol(f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params/00002000-medium1_sigma_t.vol')
IDX=sig0.shape[0]//2; sl=sig0[IDX,:,:,0]
panels=[('DRT quad (volpathsimple)',       'uni_drtquad', 500),
        ('DRT linear (volpathsimple)',     'uni_drtlin',  500),
        ('FM quad (volpathfm_sd)',         'uni_refquad', 1000),
        ('FM linear (volpathfm_linear_sd)','uni_ref',     1000)]
data=[(t,np.load(f'{ROOT}/grad_dump/{k}_mean.npy').reshape(sig0.shape)[IDX,:,:,0],n) for t,k,n in panels]
lim=np.percentile(np.abs(data[2][1]),99)
fig,axs=plt.subplots(1,4,figsize=(21,5.2))
for ax,(t,d,n) in zip(axs,data):
    im=ax.imshow(d,cmap=cubicL,vmin=-lim,vmax=lim)
    ax.set_title(f'{t}\nN={n}   <slice,σ>={(d*sl).sum():+.4e}',fontsize=10); ax.axis('off')
fig.colorbar(im,ax=axs,shrink=0.75,aspect=30,label='mean dL/dσ_t')
fig.suptitle('Uniform env, majorant_factor=2.0, 768x576, loss=sum, slice axis0@128 — original DRT shows NO linear/quad gap; FM linear is the outlier',fontsize=12)
fig.savefig(f'{ROOT}/inverse_results/fig13_drt_vs_fm_slice.png',dpi=150,bbox_inches='tight')
print('saved; projections:')
for t,d,n in data: print(f'  {t:34s} <slice,sig>={(d*sl).sum():+.4e}')
