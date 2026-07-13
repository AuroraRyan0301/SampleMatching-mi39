"""
Mesh test for prbvolpath_sm: a real triangle mesh (OBJ octahedron) INSIDE the
heterogeneous medium — exercises the surface-inside-medium path and the
segment-reset logic at opaque surface interactions.
Checks: primal parity vs prbvolpath + FD gradcheck of sigma_t grads.
"""
import os
import numpy as np
import drjit as dr
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

OBJ = os.path.join(os.path.dirname(__file__), 'octa.obj')
with open(OBJ, 'w') as f:
    f.write("""v  0.35  0.0  0.0
v -0.35  0.0  0.0
v  0.0  0.35  0.0
v  0.0 -0.35  0.0
v  0.0  0.0  0.35
v  0.0  0.0 -0.35
f 1 3 5
f 3 2 5
f 2 4 5
f 4 1 5
f 3 1 6
f 2 3 6
f 4 2 6
f 1 4 6
""")

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
        # a REAL triangle mesh INSIDE the medium (opaque diffuse): rays
        # scatter off it while inside the participating medium.
        'mesh': {
            'type': 'obj', 'filename': OBJ,
            'bsdf': {'type': 'diffuse',
                     'reflectance': {'type': 'rgb', 'value': [0.7, 0.3, 0.2]}},
        },
    })

MAXD = 6

print('== [1] mesh-in-medium: primal parity ==')
scene_ref = build({'type': 'prbvolpath', 'max_depth': MAXD})
scene_sm  = build({'type': 'prbvolpath_sm', 'max_depth': MAXD})
img_ref = np.array(mi.render(scene_ref, spp=64, seed=3))
img_sm  = np.array(mi.render(scene_sm,  spp=64, seed=3))
rel = abs(img_sm.mean() - img_ref.mean()) / img_ref.mean()
print(f'   primal mean: prb={img_ref.mean():.4f} sm={img_sm.mean():.4f} rel={rel:.3%}')
assert np.isfinite(img_sm).all() and rel < 0.05

print('== [2] mesh-in-medium: FD gradcheck of sigma_t ==')
H = 0.05
def primal_loss(gs, seed):
    sc = build({'type': 'prbvolpath', 'max_depth': MAXD}, gs)
    return float(np.array(mi.render(sc, spp=256, seed=seed)).mean())
fd = np.mean([(primal_loss(1+H, 100+s) - primal_loss(1-H, 100+s)) / (2*H) for s in range(3)])

def dirderiv(int_dict, n=6):
    sc = build(int_dict)
    p = mi.traverse(sc)
    key = [k for k in p.keys() if 'sigma_t' in k and k.endswith('.data')][0]
    vals = []
    for s in range(n):
        dr.enable_grad(p[key]); p.update()
        img = mi.render(sc, p, spp=64, seed=10+s, seed_grad=1000+s)
        dr.backward(dr.mean(img, axis=None))
        g = np.array(dr.grad(p[key])).ravel()
        dr.disable_grad(p[key])
        assert np.isfinite(g).all(), 'NaN in grads'
        vals.append(float((g * GRID.ravel()).sum()))
    return np.array(vals)

v_prb = dirderiv({'type': 'prbvolpath', 'max_depth': MAXD})
v_sm  = dirderiv({'type': 'prbvolpath_sm', 'max_depth': MAXD})
print(f'   FD  : {fd:+.5e}')
print(f'   prb : {v_prb.mean():+.5e} (std {v_prb.std():.2e})  err {abs(v_prb.mean()-fd)/abs(fd):.2%}')
print(f'   sm  : {v_sm.mean():+.5e} (std {v_sm.std():.2e})  err {abs(v_sm.mean()-fd)/abs(fd):.2%}')
ok = abs(v_sm.mean() - fd) / abs(fd) < 0.15
print('\nMESH-IN-MEDIUM ' + ('PASSED' if ok else 'FAILED'))
