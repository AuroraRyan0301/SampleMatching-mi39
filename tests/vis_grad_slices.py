"""Mid-slice visualization of gradient mean/var: mine (mi39, fixed) vs
reference (old mitsuba), following var_view_equaltime.py's style."""
import numpy as np, struct, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT='/path/to/SMmi39'
def read_vol(p):
    with open(p,'rb') as fh:
        h=fh.read(48); x,y,z,ch=struct.unpack('<4i',h[8:24])
        return np.fromfile(fh,dtype='<f4').reshape(z,y,x,ch)
CKPT=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
sig = read_vol(f'{CKPT}/00002000-medium1_sigma_t.vol').squeeze()      # (z,y,x)
R = sig.shape[0]
def gload(side,k): return np.load(f'{ROOT}/grad_dump/{side}_{k}.npy').reshape(sig.shape)
data = {s:{k:gload(s,k) for k in ('mean','std')} for s in ('mine','ref')}
for s in data: data[s]['var'] = data[s]['std']**2

AX, POS = 0, 0.5
idx = int(POS*R)
sl = lambda a: a[idx]   # axis-0 slice
dens = sl(sig)
# shared scales
mmin = min(sl(data[s]['mean']).min() for s in data); mmax = max(sl(data[s]['mean']).max() for s in data)
mabs = 0.9*max(abs(mmin),abs(mmax))
vmax = 0.9*max(sl(data[s]['var']).max() for s in data)

fig, axs = plt.subplots(3, 2, figsize=(11, 14))
for j, s in enumerate(('ref','mine')):
    name = {'ref':'reference (old mitsuba)','mine':'mine (mi39 prbvolpath_sm, fixed)'}[s]
    v = sl(data[s]['var']); m = sl(data[s]['mean'])
    im0 = axs[0,j].imshow(v, cmap='magma', vmin=0, vmax=vmax)
    axs[0,j].set_title(f'Var  —  {name}', fontsize=10)
    axs[0,j].text(0.97,0.03,f"mean={v.mean():.3e}",transform=axs[0,j].transAxes,ha='right',va='bottom',
                  fontsize=8,color='w',bbox=dict(facecolor='k',alpha=0.5))
    im1 = axs[1,j].imshow(m, cmap='RdBu_r', vmin=-mabs, vmax=mabs)
    axs[1,j].set_title(f'Mean  —  {name}', fontsize=10)
    axs[1,j].text(0.97,0.03,f"mean={m.mean():.3e}",transform=axs[1,j].transAxes,ha='right',va='bottom',
                  fontsize=8,color='k',bbox=dict(facecolor='w',alpha=0.6))
    im2 = axs[2,j].imshow(dens, cmap='viridis')
    axs[2,j].set_title('Density (ref iter-2000 state)', fontsize=10)
    for a in axs[:,j]: a.axis('off')
fig.colorbar(im0, ax=axs[0,:], shrink=0.75); fig.colorbar(im1, ax=axs[1,:], shrink=0.75); fig.colorbar(im2, ax=axs[2,:], shrink=0.75)
fig.suptitle(f'Gradient mean/var, equal-iter (N=512, view 1, loss=sum(image)) — slice axis {AX} @ {POS}', fontsize=12)
fig.savefig(f'{ROOT}/inverse_results/fig8_grad_meanvar_slices.png', dpi=150, bbox_inches='tight')
print('saved fig8')

# 数值摘要
mm, rm = data['mine']['mean'].ravel(), data['ref']['mean'].ravel()
ms, rs = data['mine']['std'].ravel(),  data['ref']['std'].ravel()
sr = sig.ravel()
print(f"global cosine(mean fields): {(mm*rm).sum()/max(np.linalg.norm(mm)*np.linalg.norm(rm),1e-30):.4f}")
print(f"global |mean| ratio (mine/ref): {np.abs(mm).mean()/np.abs(rm).mean():.3f}")
print(f"global std ratio (mine/ref): {ms.mean()/rs.mean():.3f}")
for nm,msk in [('core >0.35',sr>0.35),('mid 0.05-0.35',(sr>0.05)&(sr<=0.35)),('shell',(sr>0.005)&(sr<=0.05))]:
    c = np.corrcoef(mm[msk],rm[msk])[0,1]
    print(f"{nm:15s}: corr(mean)={c:.3f}  std_ratio={ms[msk].mean()/rs[msk].mean():.3f}")
