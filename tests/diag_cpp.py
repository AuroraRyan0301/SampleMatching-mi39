"""Bisect the lin_cpp gradient bug: primal parity vs baseline."""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=768,resy=576,parallel=False,
                   majorant_resolution_factor=8, majorant_factor=1.3,
                   use_bbox_fast_path='true',
                   envmap_filename=f'{ROOT}/data/scenes/common/uniform_white.exr',
                   medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
B={'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True}
for name,extra in [('base',{}),('cpp',{'real_interaction_cpp':True})]:
    integ=mi.load_dict({**B,**extra})
    acc=0.0
    for s in range(8):
        img=mi.render(scene,sensor=1,integrator=integ,spp=64,seed=s)
        acc+=float(np.array(img).sum())
    print(f'[{name}] primal sum (8x64spp avg) = {acc/8:.6e}',flush=True)
print('DIAG DONE')
