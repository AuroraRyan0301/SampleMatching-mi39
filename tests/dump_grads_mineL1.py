"""Equal-iter gradient statistics, user's var_view protocol: single view,
loss = dr.sum(image), Welford per-voxel mean/var. MY estimator (mi39)."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CKPT='/path/to/SMmi39/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                     resx=384, resy=288, parallel=False, majorant_resolution_factor=8,
                     medium_filename=f'{CKPT}/00002000-medium1_sigma_t.vol',
                     albedo_filename=f'{CKPT}/00002000-medium1_albedo.vol')
p = mi.traverse(scene); K='medium1.sigma_t.data'
integ = mi.load_dict({'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,
                      'probes_per_segment':1,'linear_cost':True})
N=512; SPP=8; CAM=1
mean=None; m2=None; mx=None; nonfinite=0
for it in range(N):
    seed,_ = mi.sample_tea_32(3*it+0, 0)
    seed_grad,_ = mi.sample_tea_32(3*it+1, 0)
    if seed==seed_grad: seed_grad += 100
    dr.enable_grad(p[K]); p.update()
    img = mi.render(scene, p, sensor=CAM, integrator=integ,
                    spp=SPP, spp_grad=SPP, seed=seed, seed_grad=seed_grad)
    dr.backward(dr.sum(img, axis=None))
    g = np.array(dr.grad(p[K])).ravel().astype(np.float64)
    dr.disable_grad(p[K])
    bad=~np.isfinite(g); nonfinite+=int(bad.sum()); g=np.where(bad,0,g)
    if mean is None: mean=g.copy(); m2=np.zeros_like(g); mx=np.abs(g)
    else:
        c=it+1; d=g-mean; mean+=d/c; m2+=d*(g-mean); mx=np.maximum(mx,np.abs(g))
    if it%64==0 or it==N-1:
        print(f'iter {it} (nonfinite: {nonfinite})', flush=True)
        if it>1:
            np.save(f'{ROOT}/grad_dump/mineL1_mean.npy', mean.astype(np.float32))
            np.save(f'{ROOT}/grad_dump/mineL1_std.npy', np.sqrt(m2/max(it,1)).astype(np.float32))
            np.save(f'{ROOT}/grad_dump/mineL1_max.npy', mx.astype(np.float32))
std=np.sqrt(m2/(N-1))
np.save(f'{ROOT}/grad_dump/mineL1_mean.npy', mean.astype(np.float32))
np.save(f'{ROOT}/grad_dump/mineL1_std.npy', std.astype(np.float32))
np.save(f'{ROOT}/grad_dump/mineL1_max.npy', mx.astype(np.float32))
print(f'TOTAL nonfinite: {nonfinite}')
