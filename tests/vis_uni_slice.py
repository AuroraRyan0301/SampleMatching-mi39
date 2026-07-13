"""4-panel same-slice mean-gradient comparison, uniform-light bunny protocol:
ref linear | ref quadratic | my sm | my smlin. Shared symmetric color scale."""
import numpy as np, struct, glob, re
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
from cubicl import cubicL
ROOT='/path/to/SMmi39'
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,c=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,c)
sig0=read_vol(f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params/00002000-medium1_sigma_t.vol')
Z=sig0.shape[0]; IDX=Z//2
def iters_of(tag, log_glob, pat):
    n=0
    for f in glob.glob(log_glob):
        for m in re.finditer(pat, open(f).read()): n=max(n,int(m.group(1))+1)
    return n
panels=[('ref linear (paper formal)', 'uni_ref',    iters_of('ref',f'{ROOT}/uni_refg.*.out',r'\[ref\] it (\d+)')),
        ('ref quadratic',             'uni_refquad',iters_of('rq', f'{ROOT}/uni_refquad.*.out',r'\[ref\] it (\d+)')),
        ('mine sm (quadratic)',       'uni_sm',     iters_of('sm', f'{ROOT}/uni_mine.*.out',r'\[sm\] it (\d+)')),
        ('mine smlin (linear)',       'uni_smlin',  iters_of('sl', f'{ROOT}/uni_smlin.*.out',r'\[smlin\] it (\d+)'))]
sl_sig=sig0[IDX,:,:,0]
data=[(t,np.load(f'{ROOT}/grad_dump/{k}_mean.npy').reshape(sig0.shape)[IDX,:,:,0],n) for t,k,n in panels]
lim=np.percentile(np.abs(data[0][1]),99)
fig,axs=plt.subplots(1,5,figsize=(26,5.2))
for ax,(t,d,n) in zip(axs[:4],data):
    im=ax.imshow(d,cmap=cubicL,vmin=-lim,vmax=lim)
    proj=(d*sl_sig).sum()
    ax.set_title(f'{t}\nN={n} iters   <slice,σ>={proj:+.3e}',fontsize=10); ax.axis('off')
fig.colorbar(im,ax=axs[:4],shrink=0.75,aspect=30,label='mean dL/dσ_t')
diff=data[0][1]-data[1][1]
dl=np.percentile(np.abs(diff),99)
im2=axs[4].imshow(diff,cmap=cubicL,vmin=-dl,vmax=dl)
axs[4].set_title(f'ref_linear − ref_quadratic\n<slice,σ>={(diff*sl_sig).sum():+.3e}',fontsize=10); axs[4].axis('off')
fig.colorbar(im2,ax=axs[4:],shrink=0.75,aspect=30,label='difference')
fig.suptitle(f'Uniform white env, majorant_factor=2.0, 768x576, loss=sum, slice axis0@{IDX}/256, iter-2000 state',fontsize=12)
fig.savefig(f'{ROOT}/inverse_results/fig12_uniform_4way_slice.png',dpi=150,bbox_inches='tight')
print('saved; per-panel <slice,sig>:')
for t,d,n in data: print(f'  {t:26s} N={n:4d}  {(d*sl_sig).sum():+.4e}')
