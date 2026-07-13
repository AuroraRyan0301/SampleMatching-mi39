"""
Gradient correctness check for `prbvolpath_sm`:
directional derivative of loss=mean(image) along the sigma_t grid direction,
compared three ways:
  (a) finite differences (ground truth, integrator-agnostic primal renders)
  (b) prbvolpath adjoint (established baseline estimator)
  (c) prbvolpath_sm adjoint (ours)
All three must agree in expectation. Also reports per-seed std (variance).
"""
import numpy as np
import drjit as dr
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

rng = np.random.default_rng(0)
GRID = rng.uniform(0.4, 1.6, size=(8, 8, 8, 1)).astype(np.float32)

def build(integrator_dict, grid_scale=1.0):
    return mi.load_dict({
        'type': 'scene',
        'integrator': integrator_dict,
        'light': {'type': 'constant', 'radiance': 1.0},
        'sensor': {
            'type': 'perspective', 'fov': 45,
            'to_world': mi.ScalarTransform4f().look_at([0, 0, 4], [0, 0, 0], [0, 1, 0]),
            'film': {'type': 'hdrfilm', 'width': 24, 'height': 24,
                     'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
            'sampler': {'type': 'independent', 'sample_count': 8},
        },
        'medium_box': {
            'type': 'cube', 'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'grid': mi.VolumeGrid((GRID * grid_scale).astype(np.float32)),
                    'to_world': mi.ScalarTransform4f().translate([-1, -1, -1]).scale(2.0),
                },
                'albedo': 0.8, 'scale': 2.0,
            },
        },
        'sphere': {
            'type': 'sphere', 'radius': 0.4,
            'to_world': mi.ScalarTransform4f().translate([1.6, 0, 0]),
            'bsdf': {'type': 'diffuse', 'reflectance': 0.5},
        },
    })

MAXD = 6
N_SEEDS = 8

# ---------- (a) finite differences ----------
H = 0.05
def primal_loss(grid_scale, seed):
    sc = build({'type': 'prbvolpath', 'max_depth': MAXD}, grid_scale)
    return float(np.array(mi.render(sc, spp=256, seed=seed)).mean())

fd_vals = []
for s in range(4):
    lp = primal_loss(1.0 + H, seed=100 + s)
    lm = primal_loss(1.0 - H, seed=100 + s)   # same seed: correlated FD
    fd_vals.append((lp - lm) / (2 * H))
fd = np.mean(fd_vals)
print(f'(a) FD directional derivative : {fd:+.5e}  (std {np.std(fd_vals):.2e}, n=4)')

# ---------- (b)/(c) adjoint directional derivatives ----------
def adjoint_dirderiv(int_dict):
    scene = build(int_dict)
    params = mi.traverse(scene)
    key = [k for k in params.keys() if 'sigma_t' in k and k.endswith('.data')][0]
    vals = []
    for s in range(N_SEEDS):
        dr.enable_grad(params[key]); params.update()
        img = mi.render(scene, params, spp=64, seed=10 + s, seed_grad=1000 + s)
        dr.backward(dr.mean(img, axis=None))
        g = np.array(dr.grad(params[key])).ravel()
        dr.disable_grad(params[key])
        # directional derivative along the grid-scaling direction:
        # d/dh loss(GRID*(1+h)) = sum_i g_i * GRID_i
        vals.append(float((g * GRID.ravel()).sum()))
    return np.array(vals)

v_prb = adjoint_dirderiv({'type': 'prbvolpath', 'max_depth': MAXD})
print(f'(b) prbvolpath    adjoint     : {v_prb.mean():+.5e}  (std {v_prb.std():.2e}, n={N_SEEDS})')

v_sm1 = adjoint_dirderiv({'type': 'prbvolpath_sm', 'max_depth': MAXD})
print(f'(c) prbvolpath_sm adjoint Λ=1 : {v_sm1.mean():+.5e}  (std {v_sm1.std():.2e}, n={N_SEEDS})')

v_sm4 = adjoint_dirderiv({'type': 'prbvolpath_sm', 'max_depth': MAXD, 'probes_per_segment': 4})
print(f'(d) prbvolpath_sm adjoint Λ=4 : {v_sm4.mean():+.5e}  (std {v_sm4.std():.2e}, n={N_SEEDS})')

v_lin = adjoint_dirderiv({'type': 'prbvolpath_sm', 'max_depth': MAXD, 'linear_cost': True})
print(f'(e) prbvolpath_sm linear      : {v_lin.mean():+.5e}  (std {v_lin.std():.2e}, n={N_SEEDS})')

v_lin4 = adjoint_dirderiv({'type': 'prbvolpath_sm', 'max_depth': MAXD, 'linear_cost': True,
                           'probes_per_segment': 4})
print(f'(f) prbvolpath_sm linear Λ=4  : {v_lin4.mean():+.5e}  (std {v_lin4.std():.2e}, n={N_SEEDS})')

# ---------- verdicts ----------
def rel(a, b): return abs(a - b) / max(abs(b), 1e-12)
print(f'\nrel err vs FD:  prb={rel(v_prb.mean(), fd):.2%}  sm1={rel(v_sm1.mean(), fd):.2%}  '
      f'sm4={rel(v_sm4.mean(), fd):.2%}  lin={rel(v_lin.mean(), fd):.2%}  lin4={rel(v_lin4.mean(), fd):.2%}')
ok = all(rel(v.mean(), fd) < 0.15 for v in (v_sm1, v_sm4, v_lin, v_lin4))
print('\nGRADCHECK ' + ('PASSED' if ok else 'FAILED — investigate estimator terms'))
