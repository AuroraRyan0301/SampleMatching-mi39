"""Unbiasedness test of the deferred two-stage smlin/sm on mi39:
N=500 mean <g,sigma> under the uniform-light protocol, against the
established quad anchors (refquad N=1000: -2.0718e5; DRT quad: -2.0709e5)."""
import sys, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=768,resy=576,parallel=False,
                   majorant_resolution_factor=8, majorant_factor=2.0,
                   use_bbox_fast_path=('true' if sys.argv[1]=='fp1' else 'false'),
                   envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'
sig=np.array(p[K]).ravel().astype(np.float64)
N=500; SPP=8
for name,d in [('smlin_defer_exact',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,
                               'probes_per_segment':4,'linear_cost':True}),
               ('sm_defer_exact',   {'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,
                               'probes_per_segment':4})]:
    integ=mi.load_dict(d)
    mean=None; m2=None
    for it in range(N):
        seed,_=mi.sample_tea_32(3*it+0,0); seed_grad,_=mi.sample_tea_32(3*it+1,0)
        if seed==seed_grad: seed_grad+=100
        dr.enable_grad(p[K]); p.update()
        img=mi.render(scene,p,sensor=1,integrator=integ,spp=SPP,spp_grad=SPP,
                      seed=seed,seed_grad=seed_grad)
        dr.backward(dr.sum(img,axis=None))
        g=np.array(dr.grad(p[K])).ravel().astype(np.float64)
        dr.disable_grad(p[K])
        g=np.where(np.isfinite(g),g,0)
        if mean is None: mean=g.copy(); m2=np.zeros_like(g)
        else:
            c=it+1; dlt=g-mean; mean+=dlt/c; m2+=dlt*(g-mean)
        if it%100==0: print(f'[{name}] it {it} <g,sig>={(mean*sig).sum():+.5e}',flush=True)
    std=np.sqrt(m2/(N-1))
    se=np.sqrt(((sig**2)*(std**2)).sum()/N)
    gs=(mean*sig).sum()
    np.save(f'{ROOT}/grad_dump/uni_{name}_exact_{sys.argv[1]}_mean.npy',mean.astype(np.float32))
    print(f'[{name}] DONE <g,sig>={gs:+.5e} ± {se:.2e}   '
          f'(quad anchors: refquad -2.0718e+05, DRT -2.0709e+05)',flush=True)
print('BIAS DEFER DONE')
