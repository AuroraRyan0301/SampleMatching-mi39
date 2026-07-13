"""Offline derivative-level comparison of the two estimators at the same state.
Inputs: grad_dump/{mine,ref}_{mean,std,max}.npy  (N=32 seeds each)
Reports bias (mean agreement), variance ratio, and tail behavior by region.
"""
import numpy as np, struct

ROOT = '/path/to/SMmi39'
def read_vol(p):
    with open(p, 'rb') as fh:
        h = fh.read(48); x, y, z, ch = struct.unpack('<4i', h[8:24])
        return np.fromfile(fh, dtype='<f4').reshape(z, y, x, ch)

sig = read_vol(f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/'
               f'volpathfm-linear-drt-sd-n4/params/00002000-medium1_sigma_t.vol').ravel()
mm, ms, mx_m = [np.load(f'{ROOT}/grad_dump/mine_{k}.npy') for k in ('mean', 'std', 'max')]
rm, rs, mx_r = [np.load(f'{ROOT}/grad_dump/ref_{k}.npy') for k in ('mean', 'std', 'max')]

regions = [('core  σ>0.2 ', sig > 0.2),
           ('mid   0.02-0.2', (sig > 0.02) & (sig <= 0.2)),
           ('shell 0.001-0.02', (sig > 0.001) & (sig <= 0.02)),
           ('empty ≤0.001', sig <= 0.001)]

print(f"{'region':18s} {'corr(mean)':>10s} {'|m|ratio':>9s} {'std ratio':>9s} "
      f"{'tail_m':>8s} {'tail_r':>8s} {'>10σ_m':>8s} {'>10σ_r':>8s}")
for name, msk in regions:
    a, b = mm[msk], rm[msk]
    corr = np.corrcoef(a, b)[0, 1] if msk.sum() > 2 else float('nan')
    mratio = np.abs(a).mean() / max(np.abs(b).mean(), 1e-30)
    sratio = ms[msk].mean() / max(rs[msk].mean(), 1e-30)
    # tail heaviness: per-voxel max|g| / std (Gaussian N=32 expects ~2.0-2.5)
    tm = np.median(mx_m[msk] / (ms[msk] + 1e-30))
    tr = np.median(mx_r[msk] / (rs[msk] + 1e-30))
    em = int((mx_m[msk] > 10 * (ms[msk] + 1e-30)).sum())
    er = int((mx_r[msk] > 10 * (rs[msk] + 1e-30)).sum())
    print(f'{name:18s} {corr:10.3f} {mratio:9.3f} {sratio:9.3f} '
          f'{tm:8.2f} {tr:8.2f} {em:8d} {er:8d}')

# global scale + direction agreement of the mean fields
num = float((mm * rm).sum()); den = float(np.linalg.norm(mm) * np.linalg.norm(rm))
print(f'\nglobal mean-field cosine: {num / max(den, 1e-30):.4f}')
print(f'global |mean| ratio (mine/ref): {np.abs(mm).mean() / np.abs(rm).mean():.3f}')
print(f'global std ratio  (mine/ref): {ms.mean() / rs.mean():.3f}')
print(f'global max|g|: mine {mx_m.max():.3e}  ref {mx_r.max():.3e}')
