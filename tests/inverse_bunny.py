"""
Scaled-down bunny-cloud inverse rendering on mi39: prbvolpath (baseline) vs
prbvolpath_sm. Recovers sigma_t + albedo grids from multi-view references.

Scaled down from the paper's formal config for a first validation run:
  film 192x144 (refs box-downsampled 4x from 768x576), grid fixed at 64^3
  (no multires), 400 iters, 1 random training view per iter (full frame),
  primal spp 512 / adjoint spp 8 (paper's primal_spp_factor), max_depth 16, L1 loss, Adam lr 6e-3.
Both methods share seeds, iteration schedule and hyperparameters.

Usage: python inverse_bunny.py <method>   # method in {prb, sm, smlin}
"""
import os, sys, time, json
import numpy as np
import drjit as dr
import mitsuba as mi

mi.set_variant('cuda_ad_rgb')

ROOT = '/path/to/SMmi39'
SCENE_XML = f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml'
REF_DIR = f'{ROOT}/data/mi_ref/bunny-cloud'
OUT_DIR = f'{ROOT}/inverse_results'

METHOD = sys.argv[1] if len(sys.argv) > 1 else 'prb'
INTEGRATORS = {
    'prb':   {'type': 'prbvolpath', 'max_depth': 16},
    'sm':    {'type': 'prbvolpath_sm', 'max_depth': 16, 'probes_per_segment': 4},
    'smlin': {'type': 'prbvolpath_sm', 'max_depth': 16, 'probes_per_segment': 4,
              'linear_cost': True},
}
INT_DICT = INTEGRATORS[METHOD]

DOWN = 4                 # reference downsample factor (768x576 -> 192x144)
RESX, RESY = 768 // DOWN, 576 // DOWN
N_ITER = 2000
SPP_PRIMAL = 512     # accurate primal for the L1 sign (paper: spp_grad * 64)
SPP_GRAD = 8
LR = 6e-3
GRID_RES = 64
MEDIUM_SCALE = 60.0
INIT_SIGMA_T = 0.04 / MEDIUM_SCALE
INIT_ALBEDO = 0.6
TRAIN_SENSORS = list(range(1, 64))   # 63 training views (paper convention)
TEST_SENSOR = 0                      # held-out
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
    mse = float(np.mean((a - b) ** 2))
    return -10.0 * np.log10(max(mse, 1e-12))

print(f'[{METHOD}] loading scene: {SCENE_XML}')
scene = mi.load_file(SCENE_XML, resx=RESX, resy=RESY, parallel=False)
integrator = mi.load_dict(INT_DICT)
n_sensors = len(scene.sensors())
print(f'[{METHOD}] sensors: {n_sensors}, integrator: {integrator}')

params = mi.traverse(scene)
K_SIG, K_ALB = 'medium1.sigma_t.data', 'medium1.albedo.data'
print(f'[{METHOD}] original grids:', params[K_SIG].shape, params[K_ALB].shape)

# Re-initialize both grids at 64^3 constants
params[K_SIG] = mi.TensorXf(np.full((GRID_RES,) * 3 + (1,), INIT_SIGMA_T, np.float32))
params[K_ALB] = mi.TensorXf(np.full((GRID_RES,) * 3 + (3,), INIT_ALBEDO, np.float32))
params.update()

refs = {i: load_ref(i) for i in TRAIN_SENSORS + [TEST_SENSOR]}
print(f'[{METHOD}] refs loaded: {len(refs)} views of {refs[TEST_SENSOR].shape}')

opt = mi.ad.Adam(lr=LR)
opt[K_SIG] = params[K_SIG]
opt[K_ALB] = params[K_ALB]
params.update(opt)

rng = np.random.default_rng(42)
history = []
t0 = time.time()
for it in range(N_ITER):
    view = int(rng.choice(TRAIN_SENSORS))
    ref = mi.TensorXf(refs[view].astype(np.float32))

    img = mi.render(scene, params, sensor=view, integrator=integrator,
                    spp=SPP_PRIMAL, spp_grad=SPP_GRAD,
                    seed=2 * it, seed_grad=2 * it + 1)
    loss = dr.mean(dr.abs(img - ref), axis=None)   # L1
    dr.backward(loss)
    opt.step()
    opt[K_SIG] = dr.clip(opt[K_SIG], 0.0, 250.0 / MEDIUM_SCALE)
    opt[K_ALB] = dr.clip(opt[K_ALB], 0.0, 1.0)
    params.update(opt)

    if it % 50 == 0 or it == N_ITER - 1:
        with dr.suspend_grad():
            img_test = np.array(mi.render(scene, sensor=TEST_SENSOR,
                                          integrator=integrator, spp=64, seed=999))
            img_train = np.array(mi.render(scene, sensor=PREVIEW_TRAIN,
                                           integrator=integrator, spp=64, seed=999))
        rec = {'iter': it,
               'loss': float(np.array(loss).ravel()[0]),
               'psnr_test': psnr(img_test, refs[TEST_SENSOR]),
               'psnr_train': psnr(img_train, refs[PREVIEW_TRAIN]),
               'elapsed': time.time() - t0}
        history.append(rec)
        print(f"[{METHOD}] it {it:4d} | loss {rec['loss']:.5f} | "
              f"PSNR test {rec['psnr_test']:6.2f} train {rec['psnr_train']:6.2f} | "
              f"{rec['elapsed']:7.1f}s", flush=True)

# save results
with open(f'{OUT_DIR}/{METHOD}_history.json', 'w') as f:
    json.dump(history, f, indent=1)
mi.Bitmap(np.array(mi.render(scene, sensor=TEST_SENSOR, integrator=integrator,
                             spp=256, seed=1234))).write(f'{OUT_DIR}/{METHOD}_final_test.exr')
np.save(f'{OUT_DIR}/{METHOD}_sigma_t.npy', np.array(params[K_SIG]))
print(f'[{METHOD}] DONE. final test PSNR: {history[-1]["psnr_test"]:.2f} dB, '
      f'total {time.time() - t0:.0f}s')
