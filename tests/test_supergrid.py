"""
Majorant supergrid validation:
[1] primal parity: supergrid on/off must agree statistically (both unbiased)
[2] FD gradcheck with supergrid enabled (prb + sm)
[3] perf A/B on the pathological trained grid (4 voxels at the 250 ceiling)
"""
import os, sys, time
import numpy as np, drjit as dr, mitsuba as mi

mi.set_variant(os.environ.get('MI_VARIANT', 'cuda_ad_rgb'))
ROOT = '/path/to/SMmi39'
sys.path.insert(0, f'{ROOT}/pixel_batching_mi39')
from render_batch_mi39 import render_batch

rng = np.random.default_rng(0)
GRID = rng.uniform(0.4, 1.6, size=(8, 8, 8, 1)).astype(np.float32)

def build(integrator_dict, grid, factor=0, scale=2.0):
    med = {
        'type': 'heterogeneous',
        'sigma_t': {'type': 'gridvolume', 'grid': mi.VolumeGrid(grid.astype(np.float32)),
                    'to_world': mi.ScalarTransform4f().translate([-1, -1, -1]).scale(2.0)},
        'albedo': 0.8, 'scale': scale,
    }
    if factor:
        med['majorant_resolution_factor'] = factor
        med['majorant_factor'] = 1.01
    return mi.load_dict({
        'type': 'scene', 'integrator': integrator_dict,
        'light': {'type': 'constant', 'radiance': 1.0},
        'sensor': {'type': 'perspective', 'fov': 45,
                   'to_world': mi.ScalarTransform4f().look_at([0, 0, 4], [0, 0, 0], [0, 1, 0]),
                   'film': {'type': 'hdrfilm', 'width': 32, 'height': 32,
                            'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
                   'sampler': {'type': 'independent', 'sample_count': 8}},
        'medium_box': {'type': 'cube', 'bsdf': {'type': 'null'}, 'interior': med},
        'sphere': {'type': 'sphere', 'radius': 0.35,
                   'to_world': mi.ScalarTransform4f().translate([0.5, 0, 0]),
                   'bsdf': {'type': 'diffuse',
                            'reflectance': {'type': 'rgb', 'value': [0.5, 0.5, 0.5]}}},
    })

print('== [1] primal parity: supergrid off vs on (factor 2 on an 8^3 grid) ==')
imgs = {}
for factor in (0, 2):
    sc = build({'type': 'prbvolpath', 'max_depth': 16}, GRID, factor)
    acc = [np.array(mi.render(sc, spp=128, seed=s)) for s in range(4)]
    imgs[factor] = np.mean(acc, axis=0)
rel = abs(imgs[0].mean() - imgs[2].mean()) / imgs[0].mean()
print(f'   mean: off={imgs[0].mean():.5f} on={imgs[2].mean():.5f} rel={rel:.3%}')
assert np.isfinite(imgs[2]).all() and rel < 0.02, 'primal parity FAILED'

print('== [2] FD gradcheck with supergrid enabled ==')
H = 0.05
def primal_loss(gs, seed, factor):
    sc = build({'type': 'prbvolpath', 'max_depth': 16}, GRID * gs, factor)
    return float(np.array(mi.render(sc, spp=256, seed=seed)).mean())
fd = np.mean([(primal_loss(1+H, 100+s, 2) - primal_loss(1-H, 100+s, 2)) / (2*H) for s in range(3)])
def dirderiv(idict, factor, n=6):
    sc = build(idict, GRID, factor)
    p = mi.traverse(sc)
    key = [k for k in p.keys() if 'sigma_t' in k and k.endswith('.data')][0]
    vals = []
    for s in range(n):
        dr.enable_grad(p[key]); p.update()
        img = mi.render(sc, p, spp=64, seed=10+s, seed_grad=1000+s)
        dr.backward(dr.mean(img, axis=None))
        g = np.array(dr.grad(p[key])).ravel()
        dr.disable_grad(p[key])
        assert np.isfinite(g).all()
        vals.append(float((g * GRID.ravel()).sum()))
    return np.array(vals)
for name, idict in [('prb', {'type': 'prbvolpath', 'max_depth': 16}),
                    ('sm ', {'type': 'prbvolpath_sm', 'max_depth': 16, 'probes_per_segment': 4})]:
    v = dirderiv(idict, 2)
    err = abs(v.mean() - fd) / abs(fd)
    print(f'   {name}: {v.mean():+.4e} vs FD {fd:+.4e}  err {err:.2%}  std {v.std():.2e}')
    assert err < 0.15, f'{name} gradcheck FAILED with supergrid'

print('== [3] perf A/B on the pathological grid (4 voxels at 250 ceiling) ==')
sig = np.load(f'{ROOT}/inverse_results/smlin_batched_sigma_t.npy').astype(np.float32)
DOWN = 4; RESX, RESY = 768//DOWN, 576//DOWN
def perf(factor, n=30):
    scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                         resx=RESX, resy=RESY, parallel=False,
                         majorant_resolution_factor=factor)
    p = mi.traverse(scene)
    p['medium1.sigma_t.data'] = mi.TensorXf(sig)
    p['medium1.albedo.data'] = mi.TensorXf(np.full((64,)*3+(3,), 0.6, np.float32))
    p.update()
    integ = mi.load_dict({'type': 'prbvolpath_sm', 'max_depth': 32,
                          'probes_per_segment': 4, 'linear_cost': True})
    sensors_train = dr.gather(mi.SensorPtr, scene.sensors_dr(), mi.UInt32(list(range(1, 64))))
    opt = mi.ad.Adam(lr=6e-3)
    opt['medium1.sigma_t.data'] = p['medium1.sigma_t.data']
    p.update(opt)
    ts = []
    for it in range(n):
        t0 = time.time()
        img = render_batch(8192, scene, (RESX, RESY), params=p, integrator=integ,
                           seed=1+2*it, spp=64, spp_grad=8, sensors=sensors_train)
        dr.backward(dr.mean(img, axis=None))
        opt.step(); p.update(opt); dr.sync_thread()
        ts.append(time.time() - t0)
    return np.array(ts[5:])
for factor in (0, 8):
    ts = perf(factor)
    print(f'   factor={factor}: {ts.mean():6.3f} s/iter (med {np.median(ts):.3f})', flush=True)

print('\nSUPERGRID VALIDATION PASSED')
