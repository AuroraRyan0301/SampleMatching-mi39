"""Timing ablation for smlin adjoint slowness: probes, supergrid, depth."""
import time, numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def bench(tag, mrf, integ_d):
    scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                       resx=768,resy=576,parallel=False,
                       majorant_resolution_factor=mrf, majorant_factor=2.0,
                       envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                       medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                       albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
    p=mi.traverse(scene); K='medium1.sigma_t.data'
    integ=mi.load_dict(integ_d)
    ts=[]
    for it in range(4):
        t0=time.time()
        dr.enable_grad(p[K]); p.update()
        img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=100+it)
        dr.backward(dr.sum(img,axis=None))
        dr.eval(dr.grad(p[K])); dr.sync_thread(); dr.disable_grad(p[K])
        ts.append(time.time()-t0)
    print(f'[{tag}] steady={min(ts[1:]):6.1f}s  (it0 {ts[0]:.1f}s)',flush=True)
B={'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'linear_cost':True}
bench('L4 sg8  d64 (baseline)', 8, {**B,'probes_per_segment':4})
bench('L1 sg8  d64',            8, {**B,'probes_per_segment':1})
bench('L4 sg0  d64',            0, {**B,'probes_per_segment':4})
bench('L4 sg16 d64',           16, {**B,'probes_per_segment':4})
bench('L4 sg8  d16',            8, {**B,'probes_per_segment':4,'max_depth':16})
bench('prbvolpath sg8 d64',     8, {'type':'prbvolpath','max_depth':64,'rr_depth':1064})
print('ABLATE DONE')
