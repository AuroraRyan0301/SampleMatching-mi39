"""Bunny @768x576, UNIFORM white envmap, unified majorant (supergrid x8,
factor 2.0), ref iter-2000 state. sm + smlin, N=512, SPP=8, loss=sum."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=768, resy=576, parallel=False,
                   majorant_resolution_factor=8, majorant_factor=2.0,
                   envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
sig=np.array(p[K]).ravel().astype(np.float64)
N=256; SPP=8; CAM=1
for name,d in [('prb',{'type':'prbvolpath','max_depth':64,'rr_depth':1064}),('sm',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4}),
               ('smlin',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True})]:
    integ=mi.load_dict(d)
    mean=None; m2=None; nonfinite=0
    for it in range(N):
        seed,_=mi.sample_tea_32(3*it+0,0); seed_grad,_=mi.sample_tea_32(3*it+1,0)
        if seed==seed_grad: seed_grad+=100
        dr.enable_grad(p[K]); p.update()
        img=mi.render(scene,p,sensor=CAM,integrator=integ,spp=SPP,spp_grad=SPP,
                      seed=seed,seed_grad=seed_grad)
        dr.backward(dr.sum(img,axis=None))
        g=np.array(dr.grad(p[K])).ravel().astype(np.float64)
        dr.disable_grad(p[K])
        bad=~np.isfinite(g); nonfinite+=int(bad.sum()); g=np.where(bad,0,g)
        if mean is None: mean=g.copy(); m2=np.zeros_like(g)
        else:
            c=it+1; dlt=g-mean; mean+=dlt/c; m2+=dlt*(g-mean)
        if it%64==0:
            print(f'[{name}] it {it} nf{nonfinite} <g,1>={mean.sum():+.4e} <g,sig>={(mean*sig).sum():+.4e}',flush=True)
            np.save(f'{ROOT}/grad_dump/uni_pdf_{name}_mean.npy',mean.astype(np.float32))
    np.save(f'{ROOT}/grad_dump/uni_pdf_{name}_mean.npy',mean.astype(np.float32))
    np.save(f'{ROOT}/grad_dump/uni_pdf_{name}_std.npy',np.sqrt(m2/(N-1)).astype(np.float32))
    print(f'[{name}] DONE nf{nonfinite} <g,1>={mean.sum():+.4e} <g,sig>={(mean*sig).sum():+.4e}',flush=True)
print('UNI PDF DONE')
