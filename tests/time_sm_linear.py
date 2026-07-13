"""Adjoint wall-time: quadratic vs linear at deep max_depth (the O(n^2) vs O(n) point)."""
import time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('llvm_ad_rgb')
rng = np.random.default_rng(0)
GRID = rng.uniform(1.0, 3.0, size=(8,8,8,1)).astype(np.float32)  # denser -> longer paths
def build(d):
    return mi.load_dict({'type':'scene','integrator':d,
        'light':{'type':'constant','radiance':1.0},
        'sensor':{'type':'perspective','fov':45,
            'to_world':mi.ScalarTransform4f().look_at([0,0,4],[0,0,0],[0,1,0]),
            'film':{'type':'hdrfilm','width':24,'height':24,'rfilter':{'type':'box'}},
            'sampler':{'type':'independent','sample_count':8}},
        'medium_box':{'type':'cube','bsdf':{'type':'null'},
            'interior':{'type':'heterogeneous',
                'sigma_t':{'type':'gridvolume','grid':mi.VolumeGrid(GRID),
                           'to_world':mi.ScalarTransform4f().translate([-1,-1,-1]).scale(2.0)},
                'albedo':0.9,'scale':3.0}}})
def time_adjoint(d, n=3):
    sc = build(d); p = mi.traverse(sc)
    key = [k for k in p.keys() if k.endswith('sigma_t.data')][0]
    dr.enable_grad(p[key]); p.update()
    # warmup (JIT compile)
    img = mi.render(sc, p, spp=16, seed=0, seed_grad=1); dr.backward(dr.mean(img,axis=None)); dr.sync_thread()
    t0 = time.time()
    for s in range(n):
        img = mi.render(sc, p, spp=16, seed=s+1, seed_grad=s+100)
        dr.backward(dr.mean(img, axis=None)); dr.sync_thread()
    return (time.time()-t0)/n
for md in (8, 16, 32):
    tq = time_adjoint({'type':'prbvolpath_sm','max_depth':md})
    tl = time_adjoint({'type':'prbvolpath_sm','max_depth':md,'linear_cost':True})
    print(f'max_depth={md:2d}: quadratic {tq:6.2f}s  linear {tl:6.2f}s  speedup {tq/tl:4.2f}x')
