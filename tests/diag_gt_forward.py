"""Decisive check: does mi39 forward-rendering of the GT scene match the refs?"""
import numpy as np, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb')
ROOT='/path/to/SMmi39'
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-ref-mi39.xml',
                     resx=192, resy=144, parallel=False)
integ = mi.load_dict({'type':'volpath','max_depth':64,'rr_depth':64})
def ref(i):
    b=mi.Bitmap(f'{ROOT}/data/mi_ref/bunny-cloud/ref_{i:06d}.exr').convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.Float32,False)
    a=np.array(b); return a[:576,:768].reshape(144,4,192,4,-1).mean(axis=(1,3))
for s_i in (0, 1, 32):
    img = np.array(mi.render(scene, sensor=s_i, integrator=integ, spp=512, seed=11))
    r = ref(s_i)
    mse = float(np.mean((img-r)**2))
    print(f'sensor {s_i:2d}: render mean {img.mean():.4f} ref mean {r.mean():.4f} '
          f'PSNR {-10*np.log10(mse):6.2f} relL1 {np.abs(img-r).mean()/r.mean():.3%}')
    if s_i == 0:
        mi.Bitmap(img).write(f'{ROOT}/inverse_results/diag_gt_s0.exr')
