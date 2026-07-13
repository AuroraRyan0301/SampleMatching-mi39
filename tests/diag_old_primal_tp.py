"""Old-mi3 stack: primal-only render_batch throughput at the formal recipe."""
import sys, time, numpy as np
sys.path.insert(0, '/path/to/PostTracking-opensource/python')
import drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb')
from batched import render_batch
from integrators.volpathfm_linear_sd import VolpathFMLinearSDIntegrator
ROOT='/path/to/SMmi39'
CK=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
SC='/path/to/PostTracking-opensource/data/scenes/bunny-cloud'
scene=mi.load_file(f'{SC}/bunny-cloud.xml', resx=768, resy=576,
    majorant_resolution_factor=8, majorant_factor=2.0,
    medium_filename=f'{CK}/00002000-medium1_sigma_t.vol',
    albedo_filename=f'{CK}/00002000-medium1_albedo.vol')
integ=mi.load_dict({'type':'volpathfm_linear_sd','max_depth':64,'rr_depth':1064,
                    'use_drt':True,'use_drt_subsampling':True,'use_drt_mis':False,
                    'use_nee':True,'n_samples_transmittance':4})
integ.use_nee=True
BATCH=32768; RES=mi.ScalarVector2u(768,576)
sensors=dr.gather(mi.SensorPtr, scene.sensors_dr(), mi.UInt32(list(range(1,64))))
for spp in (1024,):
    ts=[]
    for it in range(4):
        t0=time.time()
        img=render_batch(BATCH, scene, sensors, RES, integrator=integ,
                         spp=spp, seed=11+it)
        dr.eval(img); dr.sync_thread()
        ts.append(time.time()-t0)
    best=min(ts[1:])
    print(f'[old primal spp={spp}] {best:6.3f} s  ({BATCH*spp/best/1e6:6.1f} Msamp/s)',flush=True)
print('OLD PRIMAL TP DONE')
