"""One-command report PDF for Shuang Zhao / Wenzel.
Auto-upgrades: uses uni_pdf_* gradient dumps and the prb training history
if they exist, otherwise falls back to the fp1 dumps / SM-only curves."""
import json, glob, os, struct
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
sys.path.insert(0, '/path/to/SMmi39/sm_integrator_tests')
from cubicl import cubicL

ROOT = '/path/to/SMmi39'
OUT = f'{ROOT}/inverse_results/sm_mi39_report.pdf'
plt.rcParams.update({'font.size': 10, 'font.family': 'DejaVu Sans'})

pdf = PdfPages(OUT)
PAGE = (11.0, 8.5)
_pg = [0]
import textwrap
def bullets(fig, x, y, items, fs=9, width=118, dy=0.026, gap=0.012, mark='\u2022  '):
    for it in items:
        for j, ln in enumerate(textwrap.wrap(it, width)):
            fig.text(x + (0 if j == 0 else 0.016), y, (mark if j == 0 else '') + ln, fontsize=fs)
            y -= dy
        y -= gap
    return y

def save(fig):
    _pg[0] += 1
    fig.savefig(f'{ROOT}/inverse_results/report_p{_pg[0]}.png', dpi=110, bbox_inches=None)
    pdf.savefig(fig)
    plt.close(fig)

# ---------------------------------------------------------------- page 1
fig = plt.figure(figsize=PAGE)
fig.text(0.06, 0.94, 'prbvolpath_sm  —  Sample Matching volume integrator in Mitsuba 3.9',
         fontsize=17, weight='bold')
fig.text(0.06, 0.905,
         'Primal = same estimator as prbvolpath. All new machinery runs only in the adjoint pass.',
         fontsize=10.5)
fig.text(0.06, 0.845, 'Tests  (gradient test: 48-seed adjoint <g,θ> vs central FD, volpath referee. primal test: z-test vs volpath)',
         fontsize=11, weight='bold')

rows = [
 ['boundary shapes', 'box / sphere / torus  ×  quad / lin / K4', 'primal + σ grads', '18', 'all pass'],
 ['emitters', 'constant / envmap / area outside / area inside / point', 'primal + σ grads', '30', 'all pass'],
 ['media', 'homogeneous / no supergrid / albedo ramp / with surface', 'primal + σ + albedo', '21', 'all pass'],
 ['depth × RR', 'depth 16 / 64,  RR on / off', 'σ grads', '12', 'all pass'],
 ['derivative sweep', '3 shapes × 5 emitters (+ prbvolpath control column)', 'σ + albedo grads', '120', 'all pass'],
 ['phase function', 'HG g = +0.5 / −0.3', 'primal + g grads', '14', 'all pass'],
 ['light gradients', 'emitter radiance', 'grads', '7', 'pass'],
 ['empty voxels', 'carved zero-density voxels, <g,1> vs forward FD', 'σ grads, all-ones', '4', '3 pass, prb 2.3×*'],
 ['LLVM backend', 'boundary + media + depth groups', 'mirror of CUDA', '60', 'all pass'],
 ['not supported', 'light/mesh position (needs edge sampling), camera pose', 'same as prbvolpath', '—', 'documented'],
]
fig.text(0.06, 0.442, '* prbvolpath control: its detached estimator is intrinsically biased in <g,1> on '
         'dense-empty scenes (2.3× here, 1.7× bunny). Known, not an SM bug — SM is exact there.',
         fontsize=8, style='italic')
ax = fig.add_axes([0.045, 0.42, 0.91, 0.40]); ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=['group', 'scenes', 'checked', '# tests', 'result'],
               colWidths=[0.14, 0.42, 0.20, 0.06, 0.10], loc='upper center', cellLoc='left')
tbl.auto_set_font_size(False); tbl.set_fontsize(8.6); tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if r == 0: cell.set_text_props(weight='bold'); cell.set_facecolor('#f0f0f0')

fig.text(0.06, 0.415, 'Bugs the stress tests caught (all fixed + committed)', fontsize=11, weight='bold')
bugs = [
 'probe domain vs grid bbox:  Volume::eval clamps outside the grid, transport treats outside as vacuum.  '
 'Probes now clipped to grid AABB.  (was: +71% σ-grad bias on a torus boundary)',
 'dr.backward in evaluated code:  default flags delete the persistent param→texture AD edge.  '
 'Use ADFlag.ClearVertices.  (was: albedo grads correct only for the first render per process)',
 'LLVM: eager backward on unevaluated loop-exit state drops gradients.  dr.eval first.  (was: −35% albedo, LLVM only)',
 'C++ walk left mei.wi / sh_frame = 0:  anisotropic phase ran in a degenerate frame.  '
 '(was: HG transport wrong, even primal — caught by the phase test)',
 'albedo as sigma_s / max(sigma_t, eps):  ratio = 0 in empty voxels, but the scatter derivative '
 'd(sigma_t*albedo)/dsigma_t = albedo is NOT.  Use get_albedo().  (was: +69% <g,1> bias on the bunny; '
 'invisible to <g,sigma>-weighted tests — new "gones" test: zero voxels + forward FD + <g,1>)',
 'albedo-MIS density must be sigma_s, not sigma_t:  the vertex estimator replays radiance that carries '
 'an albedo factor — at albedo = 0 it has no information, yet held weight sigma_t^2/(1+sigma_t^2).  '
 '(was: zero-albedo voxels lose that share of their albedo gradient, 0.969 -> 0.999; prbvolpath: 0.904, same '
 'blind spot, no probe side to take over)',
]
bullets(fig, 0.07, 0.385, bugs, fs=8.7, width=124, dy=0.0235, gap=0.009)
save(fig)

# ---------------------------------------------------------------- page 2: gradient slices
def read_vol(p):
    with open(p, 'rb') as f:
        h = f.read(48); x, y, z, c = struct.unpack('<4i', h[8:24])
        return np.fromfile(f, dtype='<f4').reshape(z, y, x, c)

CK = f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
sig0 = read_vol(f'{CK}/00002000-medium1_sigma_t.vol')
IDX = sig0.shape[0] // 2
sl_sig = sig0[IDX, :, :, 0]

DPFX = ('uni_pdfoff' if os.path.exists(f'{ROOT}/grad_dump/uni_pdfoff_prb_std.npy')
        else 'uni_pdf')
panels = [('SM quad\nold mi3 (paper code), N=1000', 'uni_refquad'),
          ('SM linear\nold mi3, N=500', 'uni_reflin_formal'),
          ('SM quad\nmi39, N=256', f'{DPFX}_sm'),
          ('SM linear\nmi39, N=256', f'{DPFX}_smlin'),
          ('prbvolpath\nmi39, N=256', f'{DPFX}_prb')]
fig, axs = plt.subplots(3, 5, figsize=PAGE,
                        gridspec_kw={'left': 0.02, 'right': 0.90, 'top': 0.84,
                                     'bottom': 0.05, 'hspace': 0.42, 'wspace': 0.05})
means = [np.load(f'{ROOT}/grad_dump/{k}_mean.npy').reshape(sig0.shape)[IDX, :, :, 0] for _, k in panels]
stds = [np.load(f'{ROOT}/grad_dump/{k}_std.npy').reshape(sig0.shape)[IDX, :, :, 0] for _, k in panels]
# Row 1: mean, scale set by the occupied region (reference panel)
lim = np.percentile(np.abs(means[0]), 99)
for ax, (t, k), d in zip(axs[0], panels, means):
    im = ax.imshow(d, cmap=cubicL, vmin=-lim, vmax=lim)
    ax.set_title(f'{t}\n<slice,σ> = {(d*sl_sig).sum():+.3e}', fontsize=7.5)
    ax.axis('off')
fig.colorbar(im, ax=list(axs[0]), shrink=0.9, label='mean (occupied scale)')
# Row 2: SAME means, scale spanning the empty-voxel range -- where the
# all-ones projection (and prbvolpath's bias) lives.
lim2 = np.percentile(-means[-1][sl_sig == 0], 99.5)
for ax, (t, k), d in zip(axs[1], panels, means):
    im = ax.imshow(d, cmap=cubicL, vmin=-lim2, vmax=0)
    ax.set_title(f'<slice,1> = {d.sum():+.3e}', fontsize=7.5)
    ax.axis('off')
fig.colorbar(im, ax=list(axs[1]), shrink=0.9, label='mean (empty-voxel scale)')
# Row 3: per-voxel std, shared scale
slim = 100.0
s_prb = float(np.mean(stds[-1][sl_sig > 0]))
for i, (ax, (t, k), d) in enumerate(zip(axs[2], panels, stds)):
    im = ax.imshow(d, cmap=cubicL, vmin=0, vmax=slim)
    m = float(np.mean(d[sl_sig > 0]))
    extra = '' if i == len(panels) - 1 else f'   ({s_prb/m:.0f}× lower)'
    ax.set_title(f'std = {m:.1f}{extra}', fontsize=7.5)
    ax.axis('off')
fig.colorbar(im, ax=list(axs[2]), shrink=0.9, label='per-voxel std')
fv = [float(np.mean(np.load(f'{ROOT}/grad_dump/{k}_std.npy')[sig0.ravel() > 0])) for _, k in panels]
fig.text(0.02, 0.008,
         f'256³ volume, occupied voxels, std:  old quad {fv[0]:.1f} | old lin {fv[1]:.1f} | '
         f'mi39 quad {fv[2]:.1f} | lin {fv[3]:.1f} | prb {fv[4]:.1f}'
         f'  →  SM variance ÷{(fv[4]/fv[2])**2:.0f} (quad) / ÷{(fv[4]/fv[3])**2:.0f} (lin)',
         fontsize=9, weight='bold')
fig.suptitle('Same slice of dI/dσ$_t$ = backprop of sum(image), adjoint 1 per pixel.  bunny cloud, uniform white light, 768×576, spp 8.\n'
             '<slice,x> = Σ mean·x over the shown slice.  Row 1: occupied-region scale, x = σ (blind to empty voxels).  '
             'Row 2: SAME mean, colorbar\nspanning the empty-voxel range, x = 1 — prbvolpath matches row 1 but is +43% off '
             'in row 2: its bias lives in the empty voxels.', fontsize=10)
save(fig)

# ---------------------------------------------------------------- page 3: inverse rendering
fig = plt.figure(figsize=PAGE)
ax = fig.add_axes([0.07, 0.12, 0.55, 0.74])
curves = [('SM linear, analytic AABB (fp1)', 'smlin_fp1_fixed_formal', 'tab:blue'),
          ('SM linear, OptiX (fp0)', 'smlin_fp0_fixed_formal', 'tab:cyan')]
_prbh = f'{ROOT}/inverse_results/prb_fp1_fixed_formal_history.json'
if os.path.exists(_prbh) and json.load(open(_prbh))[-1]['iter'] >= 7999:
    curves.insert(0, ('prbvolpath (same recipe)', 'prb_fp1_fixed_formal', 'tab:red'))
final = []
for label, tag, col in curves:
    h = json.load(open(f'{ROOT}/inverse_results/{tag}_history.json'))
    it = [r['iter'] for r in h]; tr = [r['psnr_train'] for r in h]
    ax.plot(it, tr, color=col, label=label)
    final.append((label, h[-1]['psnr_train'], h[-1]['psnr_test'], h[-1]['elapsed'] / 3600))
ax.axhline(37.23, color='gray', ls='--', lw=1, label='paper implementation (old mi3), same scene')
ax.set_xlabel('iteration'); ax.set_ylabel('train PSNR (dB)')
ax.set_title('Inverse rendering: bunny cloud, L1, 8000 iterations, identical seeds + hyperparameters')
ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.3)
txt = 'final numbers\n\n'
for label, tr, te, hh in final:
    txt += f'{label}\n  train {tr:.2f} dB   test {te:.2f} dB   {hh:.1f} h\n\n'
txt += 'paper implementation (old mi3)\n  train 37.23 dB   (3.6 h, reference framework)\n'
fig.text(0.66, 0.75, txt, fontsize=10, va='top', family='monospace')
fig.text(0.07, 0.035, 'note: one shared lr schedule (the paper recipe) for all runs — not re-tuned per method. '
         'prbvolpath plateaus: high-variance σ gradients.', fontsize=9)
save(fig)

# ---------------------------------------------------------------- page 4: RGB old mi3 vs mi39
import mitsuba as mi
mi.set_variant('scalar_rgb')

def exr(p):
    return np.array(mi.Bitmap(p), dtype=np.float32)[..., :3]

def tm(x):
    return np.clip(x, 0, 1) ** (1 / 2.2)

def psnr(a, b):
    return -10 * np.log10(np.mean((a - b) ** 2))

MI3 = (f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/'
       'volpathfm-linear-drt-sd-n4')
gt39 = exr(f'{ROOT}/data/mi_ref/bunny-cloud/ref_000000.exr')
gt3 = exr(f'{MI3}/ref_0000.exr')
panels = [
    ('GT (test view, held out)', gt39, None),
    ('old mi3 SM linear\npaper code, 3.6 h', exr(f'{MI3}/opt_final_0000_spp_1024.exr'), gt3),
    ('mi39 SM linear, analytic AABB\nfp1, 6.7 h', exr(f'{ROOT}/inverse_results/smlin_fp1_fixed_formal_final_test.exr'), gt39),
    ('mi39 SM linear, OptiX\nfp0, 19 h', exr(f'{ROOT}/inverse_results/smlin_fp0_fixed_formal_final_test.exr'), gt39),
    ('mi39 prbvolpath\nsame recipe, 8.3 h', exr(f'{ROOT}/inverse_results/prb_fp1_fixed_formal_final_test.exr'), gt39),
]
fig = plt.figure(figsize=PAGE)
err_lim = 0.15
W, X0, GAP = 0.18, 0.01, 0.007
H = W * PAGE[0] / PAGE[1] * (576 / 768)          # keep 768x576 aspect
Y1, Y2 = 0.52, 0.52 - H - 0.045
for j, (title, img, ref) in enumerate(panels):
    x = X0 + j * (W + GAP)
    ax = fig.add_axes([x, Y1, W, H]); ax.axis('off')
    ax.imshow(tm(img))
    lab = title if ref is None else f'{title}\nPSNR {psnr(img, ref):.2f} dB'
    ax.set_title(lab, fontsize=9)
    ax2 = fig.add_axes([x, Y2, W, H]); ax2.axis('off')
    if ref is not None:
        im = ax2.imshow(np.abs(img - ref).mean(-1), cmap=cubicL, vmin=0, vmax=err_lim)
        ax2.set_title('|err|', fontsize=8)
cax = fig.add_axes([X0 + 5 * (W + GAP) + 0.005, Y2, 0.014, Y1 + H - Y2])
fig.colorbar(im, cax=cax, label=f'|error|, clipped at {err_lim}')
fig.suptitle('Inverse rendering, same held-out test view, spp 1024.  '
             'PSNR = this view vs its own GT render.', fontsize=12, y=0.93)
save(fig)

# ---------------------------------------------------------------- page 5: speed
fig = plt.figure(figsize=PAGE)
fig.text(0.06, 0.93, 'Speed (512², spp 8, seconds, A40. lower = better)', fontsize=13, weight='bold')
sp = {}
for ln in open(f'{ROOT}/stress_results/speed_cuda_ad_rgb.jsonl'):
    r = json.loads(ln)
    if r.get('status') == 'INFO':
        sp[r['cell'].split('/')[-1]] = (float(r['primal_s']), float(r['adjoint_s']))
scenes = ['blob_d16_rroff', 'blob_d64_rroff', 'blob_d64_rron', 'dense_d16_rroff', 'dense_d64_rroff', 'dense_d64_rron']
rows = []
for sc in scenes:
    row = [sc.replace('_', ' ').replace('d16', 'bounce16').replace('d64', 'bounce64')]
    for m in ('prb', 'quad', 'lin', 'k4'):
        p, a = sp[f'speed_{sc}_{m}']
        row.append(f'{p:.2f} / {a:.2f}')
    rows.append(row)
ax = fig.add_axes([0.06, 0.52, 0.88, 0.36]); ax.axis('off')
tbl = ax.table(cellText=rows, colLabels=['scene depth RR', 'prbvolpath', 'SM quad', 'SM lin', 'SM K4'],
               loc='upper center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 1.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor('#cccccc')
    if r == 0: cell.set_text_props(weight='bold'); cell.set_facecolor('#f0f0f0')
fig.text(0.06, 0.60, 'table entry = primal / adjoint seconds.   '
         'blob = 8³ sparse blob, peak σ 2, mean 0.5, mostly empty.   '
         'dense = same scene, σ = 2.5 everywhere, deep scattering.', fontsize=9)
fig.text(0.06, 0.53, 'Notes', fontsize=12, weight='bold')
bullets(fig, 0.07, 0.495, [
 'SM primal is FASTER than prbvolpath everywhere: fused C++ free-flight walk (next page).',
 'SM adjoint: lin ≈ 1.1–1.5× prbvolpath. quad / K4 pay more at high depth (more probes). RR helps a lot.',
 'bunny 768×576 spp16, per-kernel (OptiX): prb 3.86 s | quad 5.40 | K4 5.57 | lin 4.05. '
 'lin adjoint = replay 2.44 + probe flush 0.40 + one shared suffix 0.85.'], fs=9.5, width=110)
save(fig)

# ---------------------------------------------------------------- page 5: design
fig = plt.figure(figsize=PAGE)
fig.text(0.06, 0.93, 'Design notes (what is different from prbvolpath)', fontsize=13, weight='bold')
blocks = [
 ('Estimator (the paper)',
  ['σ$_t$ has two opposite gradient terms: more scatter (+) and less transmittance (−).',
   'Sample matching: evaluate BOTH at the SAME uniform probe points on each path segment.',
   'Shared points → negative correlation → much lower variance. Unbiased.',
   'Variants: quad = one radiance suffix per segment (O(n²)).  lin = one suffix per path, '
   'reservoir-picked (O(n)).  K4 = 4-slot striped reservoir, exact for paths ≤ 4 segments.']),
 ('Deferred probes (two-stage adjoint)',
  ['Problem: probes trace rays. Rays inside the big replay kernel → OptiX saves 128 registers '
   '+ 3 KB/thread around every optixTrace. Very slow.',
   'Fix: replay loop only APPENDS a 68-byte record per segment (dr.scatter_inc, compacted).',
   'After the loop: read the counter, gather records, run all probes in one small separate kernel.',
   'Result: SM linear adjoint 26.5 s → 4.1 s on the bunny.']),
 ('Fused C++ free-flight walk (Medium::sample_real_interaction_fused)',
  ['One C++ loop fuses the majorant-supergrid DDA and accept-until-real null tracking.',
   'The Python loop body only ever sees a REAL scatter or an escape.',
   'Null collisions never spin the fat kernel → warp stays aligned on the expensive body.',
   'This is also why SM primal beats prbvolpath primal.',
   'Careful: fill the WHOLE MediumInteraction (wi, sh_frame, wavelengths, time, majorant…).']),
 ('Reservoir without color bias',
  ['DRT-style reservoir picks with mean-of-ratios → small color bias in colored media.',
   'Ours: scalar selection weight v = mean(w), additive → pick probability exactly v/V.',
   'Compensate per channel: w · V / v.  Bias gone (measured 35% → 0.02%).']),
 ('Dr.Jit interaction of the deferred (evaluated-mode) adjoint',
  ['The deferred flush calls dr.backward outside any symbolic loop → uses flags=ADFlag.ClearVertices '
   'to keep the persistent param→texture edges alive across renders.',
   'Loop outputs feeding the linear suffix are dr.eval-ed before its backward '
   '(required on the LLVM backend).']),
]
y = 0.88
for title, lines in blocks:
    fig.text(0.06, y, title, fontsize=11, weight='bold'); y -= 0.033
    y = bullets(fig, 0.08, y, lines, fs=9, width=120, dy=0.026, gap=0.004, mark='\u2013  ')
    y -= 0.012
save(fig)

pdf.close()
print('WROTE', OUT)
