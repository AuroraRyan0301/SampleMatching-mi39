"""<g,1> vs majorant setting: does the AD mean depend on majorant headroom?"""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def load(mrf,mf):
    return mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39-neeoff.xml',
        resx=768, resy=576, parallel=False,
        majorant_resolution_factor=mrf, majorant_factor=mf,
        envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
        medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
        albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
CFG=[(8,1.01),(8,1.3),(8,2.0),(8,4.0),(0,2.0)]
INT=[('prb',{'type':'prbvolpath','max_depth':64,'rr_depth':1064}),
     ('smq',{'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4})]
for mrf,mf in CFG:
    scene=load(mrf,mf); p=mi.traverse(scene); K='medium1.sigma_t.data'
    for nm,d in INT:
        integ=mi.load_dict(d); tot=0.; N=12
        for it in range(N):
            seed,_=mi.sample_tea_32(3*it+0,0); seed_grad,_=mi.sample_tea_32(3*it+1,0)
            if seed==seed_grad: seed_grad+=100
            dr.enable_grad(p[K]); p.update()
            img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=seed,seed_grad=seed_grad)
            dr.backward(dr.sum(img,axis=None))
            g=np.array(dr.grad(p[K])).ravel().astype(np.float64)
            dr.disable_grad(p[K]); tot+=np.where(np.isfinite(g),g,0).sum()
        print(f'[res={mrf} factor={mf} {nm}] <g,1> = {tot/N:+.6e}',flush=True)
print('MAJ G1 DONE')
