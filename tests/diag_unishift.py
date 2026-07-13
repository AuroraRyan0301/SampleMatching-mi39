import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def load(xml):
    return mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/{xml}',
        resx=768, resy=576, parallel=False,
        majorant_resolution_factor=8, majorant_factor=2.0,
        envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
        medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
        albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
for xml in ('bunny-cloud-mi39.xml','bunny-cloud-mi39-neeoff.xml'):
    scene=load(xml); p=mi.traverse(scene); K='medium1.sigma_t.data'
    vp=mi.load_dict({'type':'volpath','max_depth':64,'rr_depth':1064})
    img=mi.render(scene,sensor=1,integrator=vp,spp=64,seed=7)
    print(f'[{xml}] primal_sum={np.array(img).sum():+.6e}',flush=True)
    integ=mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
    N=8; tot=0.
    for it in range(N):
        seed,_=mi.sample_tea_32(3*it+0,0); seed_grad,_=mi.sample_tea_32(3*it+1,0)
        if seed==seed_grad: seed_grad+=100
        dr.enable_grad(p[K]); p.update()
        img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=seed,seed_grad=seed_grad)
        dr.backward(dr.sum(img,axis=None))
        g=np.array(dr.grad(p[K])).ravel().astype(np.float64)
        dr.disable_grad(p[K]); tot+=np.where(np.isfinite(g),g,0).sum()
    print(f'[{xml}] prb <g,1> (N={N}) = {tot/N:+.6e}',flush=True)
print('UNISHIFT DONE')
