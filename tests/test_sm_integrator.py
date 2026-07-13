"""
Smoke test for the built-in `prbvolpath_sm` integrator (source-built mi39 tree).
1. registration + primal parity vs prbvolpath (statistical)
2. adjoint: sigma_t grid gradients finite & nonzero
3. probes_per_segment=4
4. surface branch: scene contains a diffuse sphere
"""
import numpy as np
import drjit as dr
import mitsuba as mi
mi.set_variant('llvm_ad_rgb')

rng = np.random.default_rng(0)
GRID = rng.uniform(0.4, 1.6, size=(8, 8, 8, 1)).astype(np.float32)

def build(integrator_dict):
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
        # heterogeneous medium in a null box
        'medium_box': {
            'type': 'cube',
            'bsdf': {'type': 'null'},
            'interior': {
                'type': 'heterogeneous',
                'sigma_t': {
                    'type': 'gridvolume',
                    'grid': mi.VolumeGrid(GRID),
                    'to_world': mi.ScalarTransform4f().translate([-1, -1, -1]).scale(2.0),
                },
                'albedo': 0.8,
                'scale': 2.0,
            },
        },
        # a diffuse sphere to exercise the surface branch
        'sphere': {
            'type': 'sphere', 'radius': 0.4,
            'to_world': mi.ScalarTransform4f().translate([1.6, 0, 0]),
            'bsdf': {'type': 'diffuse', 'reflectance': 0.5},
        },
    })

print('== [1] registration & primal parity ==')
scene_ref = build({'type': 'prbvolpath', 'max_depth': 6})
scene_sm  = build({'type': 'prbvolpath_sm', 'max_depth': 6})
print('   loaded:', scene_sm.integrator())
img_ref = np.array(mi.render(scene_ref, spp=32, seed=3))
img_sm  = np.array(mi.render(scene_sm,  spp=32, seed=3))
m_ref, m_sm = img_ref.mean(), img_sm.mean()
rel = abs(m_sm - m_ref) / m_ref
print(f'   primal mean: prbvolpath={m_ref:.4f}  prbvolpath_sm={m_sm:.4f}  rel diff={rel:.3%}')
assert np.isfinite(img_sm).all(), 'NaN/inf in SM primal render'
assert rel < 0.05, 'primal renders diverge by more than 5%'

print('== [2] adjoint: sigma_t grid gradients ==')
params = mi.traverse(scene_sm)
key = [k for k in params.keys() if 'sigma_t' in k and k.endswith('.data')][0]
print('   key:', key)
dr.enable_grad(params[key]); params.update()
img = mi.render(scene_sm, params, spp=16, seed=7, seed_grad=8)
loss = dr.mean(img, axis=None)
dr.backward(loss)
g = np.array(dr.grad(params[key])).ravel()
nz = (g != 0).sum()
print(f'   grad: shape={g.shape} nonzero={nz}/{g.size} range=[{g.min():.3e}, {g.max():.3e}]')
assert np.isfinite(g).all(), 'NaN/inf in gradients'
assert nz > g.size * 0.5, 'too few nonzero gradient entries'

print('== [3] gradient sign sanity (darker when denser => mostly negative) ==')
print(f'   negative fraction: {(g < 0).mean():.2%}')

print('== [4] probes_per_segment = 4 ==')
scene_l4 = build({'type': 'prbvolpath_sm', 'max_depth': 6, 'probes_per_segment': 4})
params4 = mi.traverse(scene_l4)
key4 = [k for k in params4.keys() if 'sigma_t' in k and k.endswith('.data')][0]
dr.enable_grad(params4[key4]); params4.update()
img4 = mi.render(scene_l4, params4, spp=16, seed=7, seed_grad=8)
dr.backward(dr.mean(img4, axis=None))
g4 = np.array(dr.grad(params4[key4])).ravel()
print(f'   grad(Λ=4): nonzero={(g4 != 0).sum()}/{g4.size} range=[{g4.min():.3e}, {g4.max():.3e}]')
assert np.isfinite(g4).all()

print('\nALL prbvolpath_sm SMOKE TESTS PASSED')
