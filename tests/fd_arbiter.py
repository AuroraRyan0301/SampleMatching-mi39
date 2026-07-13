import numpy as np, struct, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb'); dr.set_thread_count(8)
ROOT='/path/to/SMmi39'
CKPT=f'{ROOT}/reference_run_output/bunny-cloud-l1-6e-3-formal-local-single-gpu/volpathfm-linear-drt-sd-n4/params'
def read_vol(p):
    with open(p,'rb') as f:
        h=f.read(48);x,y,z,ch=struct.unpack('<4i',h[8:24]);return np.fromfile(f,dtype='<f4').reshape(z,y,x,ch)
sig0=read_vol(f'{CKPT}/00002000-medium1_sigma_t.vol')
alb =read_vol(f'{CKPT}/00002000-medium1_albedo.vol')
scene=mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml',
                   resx=384,resy=288,parallel=False,majorant_resolution_factor=8,
                   medium_filename=f'{CKPT}/00002000-medium1_sigma_t.vol',
                   albedo_filename=f'{CKPT}/00002000-medium1_albedo.vol')
p=mi.traverse(scene); K='medium1.sigma_t.data'; integ=mi.load_dict({'type':'prbvolpath','max_depth':64,'rr_depth':1064})
H=0.02; CAM=1; SPP=4096
def loss(scale):
    p[K]=mi.TensorXf((sig0*scale).astype(np.float32)); p.update()
    return float(np.array(mi.render(scene,sensor=CAM,integrator=integ,spp=SPP,seed=7)).sum())
# multiplicative FD: d/dh loss(sigma*(1+h)) = <grad, sigma>
fd=(loss(1+H)-loss(1-H))/(2*H)
print(f'FD <dLoss/dsigma, sigma> = {fd:+.4e}  (spp {SPP}, H {H})')
