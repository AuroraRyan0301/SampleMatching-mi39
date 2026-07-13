# Sample Matching on Mitsuba 3.9

Port of the **Sample Matching** differentiable volume rendering estimator
(Yu et al., ACM TOG 2026) to **Mitsuba 3.9.0**, as a built-in AD integrator
`prbvolpath_sm` (based on `prbvolpath`), plus the test/validation harness,
pixel-batched training scripts, and results.

The Mitsuba 3 source changes live on a branch of the fork:
**https://github.com/AuroraRyan0301/mitsuba3/tree/sample-matching**
(21 commits on top of `v3.9.0`: integrator, `Medium::get_albedo`, majorant
supergrid DDA, C++ real-interaction walks, deferred two-stage adjoint,
K=4 segment reservoir, correctness fixes).

## Integrator variants (one plugin)

| config | cost | description |
|---|---|---|
| `prbvolpath_sm` (quad) | O(n²) | probes + one radiance suffix per segment |
| `+ linear_cost=True` (lin) | O(n) | one reservoir-picked suffix per path |
| `+ segment_reservoir=4` (K4) | bounded | 4-slot striped reservoir, exact for ≤4 segments |

Production defaults: `defer_probes=True` (two-stage adjoint; probes run in
small dedicated kernels, not inside the OptiX megakernel) and
`real_interaction_fused=True` (C++ walk fusing the majorant-supergrid DDA
with accept-until-real null tracking; primal is faster than stock
`prbvolpath` because null collisions never spin the fat kernel).

## Environment

- Linux, NVIDIA A40 (CUDA 12.8), OptiX via the NVIDIA driver
- gcc 13.3.1, cmake 3.26.5
  (cmake ≥ 4 breaks Mitsuba submodules — use 3.26.x with the flag below)
- conda env: python 3.11, ninja, numpy, pytest, matplotlib
  (`environment/mi39-dev.yml`)

### Build

```bash
conda env create -f environment/mi39-dev.yml && conda activate mi39-dev
git clone --recursive -b sample-matching https://github.com/AuroraRyan0301/mitsuba3.git
cd mitsuba3 && mkdir build && cd build
cmake -GNinja -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DMI_DEFAULT_VARIANTS="scalar_rgb;llvm_ad_rgb;cuda_ad_rgb" ..
ninja
source setpath.sh
```

`environment/build_mi39.sh` scripts the same steps.
CPU (`llvm_ad_rgb`) needs `DRJIT_LIBLLVM_PATH` pointed at a libLLVM shared library.
In-tree `.py` edits need no rebuild, but note that `source setpath.sh` imports the
*copies* under `build/python/mitsuba/python/ad/integrators/` — re-sync them after
editing `src/python/python/ad/integrators/`.

## Tests

`tests/stress_sm.py <group> [variant]` — one structured `CELL` line per test,
JSONL appended to `stress_results/`. Scripts assume the paths at their top
(`ROOT = /path/to/SMmi39`) are adjusted to your checkout. Groups:

| group | contents |
|---|---|
| `boundary` | box / sphere / non-convex torus medium boundaries, primal parity + σ gradients |
| `emitter` | constant / envmap / area (outside + inside the medium) / point |
| `media` | homogeneous, no-supergrid, gradient-ramp albedo, embedded surfaces |
| `depthrr` | depth 16/64 × Russian roulette on/off |
| `gradfull` | 3 boundaries × 5 emitters × {σ, albedo} × {quad, lin, K4} + prbvolpath control (shard with the `GF_BOUNDARY` env var) |
| `gphase` | Henyey–Greenstein g = ±, primal + dL/dg |
| `gones` | grid with exact-zero voxels, ⟨g,1⟩ vs **forward** FD (regression guard for empty-voxel gradient bugs; prbvolpath emitted as a documented-bias control) |
| `lightsgrad`, `camera`, `speed` | emitter radiance grads, camera support status, primal/adjoint timing tables |

Methodology: gradient cells compare the 48-seed adjoint ⟨g,dir⟩ against a
finite-difference referee (stock `volpath`, spp 2048, correlated seeds).
Two pitfalls the harness encodes (both cost us real debugging time):
never central-difference across the σ ≥ 0 clamp (use `fd_forward=True`),
and never rely only on ⟨g,σ⟩-weighted projections (blind to empty voxels).

Final state: all groups pass (gradfull 120/120); `prbvolpath` itself is
~1.7–2.3× biased in the ⟨g,1⟩ projection on dense–empty scenes (documented
WARN control — the bias Sample Matching removes).

## Training (pixel batching)

`tests/inverse_bunny_formal.py <method> <fp>` — the paper's bunny-cloud
recipe (8000 iters, batch 32768 pixels over 63 views, L1, coarse-to-fine).
`fp1` = analytic AABB fast path (fast), `fp0` = OptiX.
`pixel_batching/render_batch_mi39.py` is the ray-centric `render_batch`
(pure Python `dr.CustomOp`; no kernel changes needed on mi39).

| run | train PSNR | test PSNR | wall clock (A40) |
|---|---|---|---|
| SM linear, mi39, fp1 (analytic) | 37.94 dB | 35.56 dB | 6.7 h |
| SM linear, mi39, fp0 (OptiX) | 37.97 dB | 35.67 dB | 19.0 h |
| prbvolpath, mi39, fp1 (analytic), same recipe | 25.30 dB | 24.63 dB | 8.3 h |
| SM linear, original mi3 paper code | 37.23 dB | — | 3.6 h |

fp0 vs fp1 is the ray-intersection path, orthogonal to the integrator.
fp0 = OptiX, which is what stock Mitsuba always uses on CUDA variants;
fp1 = our analytic-AABB scene fast path (single-box scenes only; present on
the fork branch as an opt-in flag `use_bbox_fast_path`, default off, and not
intended for the eventual upstream PR) that bypasses OptiX for both
integrators. The stock-Mitsuba comparison is therefore the fp0 column. Measured per training step (batch 32768, primal spp 1024,
adjoint spp 16, final 256^3 state):

| step phase | fp1 SM lin | fp1 prbvolpath | fp0 SM lin | fp0 prbvolpath |
|---|---|---|---|---|
| primal | 1.93 s | 3.21 s | 4.68 s | 8.63 s |
| adjoint | 0.55 s | 0.52 s | 0.90 s | 0.70 s |
| step | 2.48 s | 3.72 s | 5.57 s | 9.32 s |

Adjoint cost is on par with prbvolpath in both modes; the wall-clock gaps are
all primal-side (training renders 64x more primal than adjoint samples), where
the fused C++ interaction walk wins — and wins more under OptiX, since fewer
loop iterations mean fewer trace calls. The old-mi3 pipeline's remaining edge
is primal-transport throughput of its box-specialized integrator (34.9 vs 17.4
Msamples/s), not the SM adjoint.

## Results

- `results/sm_mi39_report.pdf` — full validation & stress-test report
  (test matrix, bugs found + fixes, dI/dσ mean/variance slices vs the
  original implementation, inverse-rendering comparison, per-kernel timing).
  Page PNGs in `results/report_pages/`.
- `results/stress_results/*.jsonl` — raw test records.
- `results/*_history.json` — training curves for the table above.

Large data (reference view renders, volume checkpoints, paper scenes) is
not in this repo; scene XMLs and the small stress-test assets are under
`scenes/`.
