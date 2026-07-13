"""
Bunny-cloud inverse rendering on mi39 with PIXEL BATCHING (the paper's
training scheme): each Adam step samples `BATCH` random (sensor, pixel)
pairs across all 63 training views via render_batch, so the volume is
constrained from every direction simultaneously.

prbvolpath (baseline) vs prbvolpath_sm, identical seeds/hyperparameters.
Usage: python inverse_bunny_batched.py <method>   # prb | sm | smlin
"""
import os, sys, time, json
import numpy as np
import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')
# nanothread's pool defaults to every hyperthread on the machine; heavy
# oversubscription aggravates a known drjit-core race (drjit issue #180,
# nanothread queue.cpp `remain == 1`). Cap the pool.
dr.set_thread_count(8)

ROOT = '/path/to/SMmi39'
sys.path.insert(0, f'{ROOT}/pixel_batching_mi39')
from render_batch_mi39 import render_batch

SCENE_XML = f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml'
REF_DIR = f'{ROOT}/data/mi_ref/bunny-cloud'
OUT_DIR = f'{ROOT}/inverse_results'

METHOD = sys.argv[1] if len(sys.argv) > 1 else 'prb'
FP = sys.argv[2] if len(sys.argv) > 2 else 'fp1'   # fp1=analytic AABB, fp0=OptiX
TAG = f'{METHOD}_{FP}_fixed'                       # fresh outputs: fixed reservoir + defer + majorant 2.0
# Paper formal configs: rr_depth = max_depth + 1000 (Russian roulette OFF,
# opt_config.py:157) and use_drt_mis=False ("pure matched, no MIS-mix" —
# volpathfm-[linear-]drt-sd-n4, the configs behind the 37.26/38.91 dB figure).
INTEGRATORS = {
    'prb':   {'type': 'prbvolpath', 'max_depth': 64, 'rr_depth': 1064},
    'sm':    {'type': 'prbvolpath_sm', 'max_depth': 64, 'rr_depth': 1064,
              'probes_per_segment': 4, 'use_probe_mis': False},
    'smlin': {'type': 'prbvolpath_sm', 'max_depth': 64, 'rr_depth': 1064,
              'probes_per_segment': 4, 'use_probe_mis': False,
              'linear_cost': True},
}
INT_DICT = INTEGRATORS[METHOD]

DOWN = 1                 # full 768x576, as in the paper's formal runs
RESX, RESY = 768 // DOWN, 576 // DOWN       # 192 x 144
N_ITER = 8000
BATCH = 32768                                # paper: batch_size 32768
SPP_PRIMAL = 1024                            # paper: spp_grad * primal_spp_factor(64)
SPP_GRAD = 16
LR = 6e-3
GRID_RES = 64            # starting resolution
# Coarse-to-fine: 64^3 -> 128^3 -> 256^3, matching the paper's absolute warmup
# (80/160 iters of 8000 = its upsample=[0.01,0.02]); lr halves at each event.
UPSAMPLE_AT = {80: 128, 160: 256}
MEDIUM_SCALE = 60.0
INIT_SIGMA_T = 0.04 / MEDIUM_SCALE
INIT_ALBEDO = 0.6
TRAIN_SENSORS = list(range(1, 64))
TEST_SENSOR = 0
PREVIEW_TRAIN = 1

os.makedirs(OUT_DIR, exist_ok=True)

def downsample(img, f):
    h, w = img.shape[0] // f * f, img.shape[1] // f * f
    return img[:h, :w].reshape(h // f, f, w // f, f, -1).mean(axis=(1, 3))

def load_ref(idx):
    bmp = mi.Bitmap(os.path.join(REF_DIR, f'ref_{idx:06d}.exr'))
    bmp = bmp.convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.Float32, False)
    return downsample(np.array(bmp), DOWN)

def psnr(a, b):
    return -10.0 * np.log10(max(float(np.mean((a - b) ** 2)), 1e-12))

print(f'[{METHOD}] loading scene')
scene = mi.load_file(SCENE_XML, resx=RESX, resy=RESY, parallel=False,
                     majorant_resolution_factor=8, majorant_factor=2.0,
                     use_bbox_fast_path=('true' if FP == 'fp1' else 'false'))
integrator = mi.load_dict(INT_DICT)

params = mi.traverse(scene)
K_SIG, K_ALB = 'medium1.sigma_t.data', 'medium1.albedo.data'
params[K_SIG] = mi.TensorXf(np.full((GRID_RES,) * 3 + (1,), INIT_SIGMA_T, np.float32))
params[K_ALB] = mi.TensorXf(np.full((GRID_RES,) * 3 + (3,), INIT_ALBEDO, np.float32))
params.update()

# References: stacked in TRAIN_SENSORS order for direct sensor_idx lookup,
# flattened per channel for dr.gather.
refs_np = np.stack([load_ref(i) for i in TRAIN_SENSORS])          # (63,H,W,3)
refs_flat = [mi.Float(np.ascontiguousarray(refs_np[..., c]).ravel())
             for c in range(3)]
ref_test, ref_train_prev = load_ref(TEST_SENSOR), load_ref(PREVIEW_TRAIN)
print(f'[{METHOD}] refs: {refs_np.shape}')

# Training-sensor subset as a gatherable SensorPtr array
sensors_train = dr.gather(mi.SensorPtr, scene.sensors_dr(),
                          mi.UInt32(TRAIN_SENSORS))
print(f'[{METHOD}] train sensors: {dr.width(sensors_train)}, integrator: {integrator}')

opt = mi.ad.Adam(lr=LR)
opt[K_SIG] = params[K_SIG]
opt[K_ALB] = params[K_ALB]
params.update(opt)

history = []
CKPT_SIG = f'{OUT_DIR}/{TAG}_formal_ckpt_sigma.npy'
CKPT_ALB = f'{OUT_DIR}/{TAG}_formal_ckpt_albedo.npy'
CKPT_IT  = f'{OUT_DIR}/{TAG}_formal_ckpt_iter.txt'
START_IT = 0
if os.path.exists(CKPT_IT):
    START_IT = int(open(CKPT_IT).read().strip()) + 1
    sig_np, alb_np = np.load(CKPT_SIG), np.load(CKPT_ALB)
    opt[K_SIG] = mi.TensorXf(sig_np)   # Adam moments restart (acceptable for repro)
    opt[K_ALB] = mi.TensorXf(alb_np)
    params.update(opt)
    hist_p = f'{OUT_DIR}/{TAG}_formal_history.json'
    if os.path.exists(hist_p):
        history = json.load(open(hist_p))
    print(f'[{METHOD}] RESUMED from iter {START_IT} (grid {sig_np.shape})', flush=True)

t0 = time.time()
for it in range(START_IT, N_ITER):
    # --- coarse-to-fine upsampling (resets Adam moments for the resized keys)
    if it in UPSAMPLE_AT and opt[K_SIG].shape[0] < UPSAMPLE_AT[it]:
        n = UPSAMPLE_AT[it]
        opt[K_SIG] = dr.resample(dr.detach(opt[K_SIG]), shape=(n, n, n, 1))
        opt[K_ALB] = dr.resample(dr.detach(opt[K_ALB]), shape=(n, n, n, 3))
        params.update(opt)
        print(f'[{METHOD}] it {it}: upsampled grids to {n}^3', flush=True)

    img, sidx, pix = render_batch(BATCH, scene, (RESX, RESY), params=params,
                                  integrator=integrator, seed=1 + 2 * it,
                                  spp=SPP_PRIMAL, spp_grad=SPP_GRAD,
                                  sensors=sensors_train, return_coords=True)
    # Gather matching reference pixel values
    px = dr.minimum(mi.UInt32(pix.x), RESX - 1)
    py = dr.minimum(mi.UInt32(pix.y), RESY - 1)
    lin = (sidx * RESY + py) * RESX + px
    ref_rgb = mi.Vector3f(*[dr.gather(mi.Float, refs_flat[c], lin) for c in range(3)])
    ref_img = mi.TensorXf(dr.ravel(ref_rgb), shape=(1, BATCH, 3))

    loss = dr.mean(dr.abs(img - ref_img), axis=None)   # L1 over the batch
    dr.backward(loss)
    # lr halves after each upsampling event (paper's upsample_lr_factor);
    # albedo trains at 2x the sigma_t lr (reference param_lr_factors).
    f = 0.5 ** sum(1 for u in UPSAMPLE_AT if it >= u)
    opt.set_learning_rate({K_SIG: LR * f, K_ALB: 2.0 * LR * f})
    opt.step()
    opt[K_SIG] = dr.clip(opt[K_SIG], 0.0, 250.0)  # reference clamps raw data to max_density
    opt[K_ALB] = dr.clip(opt[K_ALB], 0.0, 1.0)
    params.update(opt)

    if it % 250 == 0 or it == N_ITER - 1:
        with dr.suspend_grad():
            img_test = np.array(mi.render(scene, sensor=TEST_SENSOR,
                                          integrator=integrator, spp=256, seed=999))
            img_train = np.array(mi.render(scene, sensor=PREVIEW_TRAIN,
                                           integrator=integrator, spp=256, seed=999))
        rec = {'iter': it,
               'loss': float(np.array(loss).ravel()[0]),
               'psnr_test': psnr(img_test, ref_test),
               'psnr_train': psnr(img_train, ref_train_prev),
               'sigma_mean': float(np.array(opt[K_SIG]).mean()),
               'elapsed': time.time() - t0}
        history.append(rec)
        np.save(CKPT_SIG, np.array(opt[K_SIG]))
        np.save(CKPT_ALB, np.array(opt[K_ALB]))
        with open(f'{OUT_DIR}/{TAG}_formal_history.json', 'w') as fh:
            json.dump(history, fh, indent=1)
        open(CKPT_IT, 'w').write(str(it))
        print(f"[{METHOD}] it {it:4d} | loss {rec['loss']:.5f} | "
              f"PSNR test {rec['psnr_test']:6.2f} train {rec['psnr_train']:6.2f} | "
              f"sig {rec['sigma_mean']:.4f} | {rec['elapsed']:7.1f}s", flush=True)

with open(f'{OUT_DIR}/{TAG}_formal_history.json', 'w') as f:
    json.dump(history, f, indent=1)
mi.Bitmap(np.array(mi.render(scene, sensor=TEST_SENSOR, integrator=integrator,
                             spp=1024, seed=1234))).write(f'{OUT_DIR}/{TAG}_formal_final_test.exr')
np.save(f'{OUT_DIR}/{TAG}_formal_sigma_t.npy', np.array(params[K_SIG]))
print(f'[{METHOD}] DONE. final test PSNR {history[-1]["psnr_test"]:.2f} dB, '
      f'{time.time() - t0:.0f}s')
