import numpy as np, drjit as dr, mitsuba as mi, os
mi.set_variant('cuda_ad_rgb')
ROOT='/path/to/SMmi39'
scene = mi.load_file(f'{ROOT}/data/scenes/bunny-cloud/bunny-cloud-mi39.xml', resx=192, resy=144, parallel=False)
integ = mi.load_dict({'type':'prbvolpath','max_depth':16})
p = mi.traverse(scene)
KS,KA='medium1.sigma_t.data','medium1.albedo.data'
p[KS]=mi.TensorXf(np.full((64,64,64,1),0.04/60,np.float32))
p[KA]=mi.TensorXf(np.full((64,64,64,3),0.6,np.float32)); p.update()
def ref(i):
    b=mi.Bitmap(f'{ROOT}/data/mi_ref/bunny-cloud/ref_{i:06d}.exr').convert(mi.Bitmap.PixelFormat.RGB,mi.Struct.Type.Float32,False)
    a=np.array(b); h,w=144*4,192*4; return a[:h,:w].reshape(144,4,192,4,-1).mean(axis=(1,3))
r0=ref(0)
img0=np.array(mi.render(scene,sensor=0,integrator=integ,spp=128,seed=7))
print('init sensor0: render mean %.4f  ref mean %.4f'%(img0.mean(),r0.mean()))
print('  bright-sky rows (top 20): render %.4f ref %.4f'%(img0[:20].mean(),r0[:20].mean()))
d=np.abs(img0-r0); print('  L1 map: mean %.4f  p99 %.3f  argmax_rowcol'%(d.mean(),np.percentile(d,99)), np.unravel_index(d.argmax(),d.shape))
mi.Bitmap(img0).write(f'{ROOT}/inverse_results/diag_init_s0.exr'); mi.Bitmap(np.ascontiguousarray(r0)).write(f'{ROOT}/inverse_results/diag_ref_s0.exr')
# --- 10 Adam steps, watch param stats ---
opt=mi.ad.Adam(lr=6e-3); opt[KS]=p[KS]; opt[KA]=p[KA]; p.update(opt)
rng=np.random.default_rng(42)
for it in range(10):
    v=int(rng.choice(range(1,64))); rt=mi.TensorXf(ref(v).astype(np.float32))
    img=mi.render(scene,p,sensor=v,integrator=integ,spp=512,spp_grad=8,seed=2*it,seed_grad=2*it+1)
    loss=dr.mean(dr.abs(img-rt),axis=None); dr.backward(loss)
    gs=np.array(dr.grad(opt[KS])); ga=np.array(dr.grad(opt[KA]))
    opt.step(); opt[KS]=dr.clip(opt[KS],0.0,250.0/60); opt[KA]=dr.clip(opt[KA],0.0,1.0); p.update(opt)
    s=np.array(opt[KS]); a=np.array(opt[KA])
    print(f'it{it}: view{v:2d} loss {float(np.array(loss).ravel()[0]):.4f} | grad_s[mean{gs.mean():+.2e} max|{np.abs(gs).max():.2e}] '
          f'| sigma[data mean {s.mean():.4f} max {s.max():.3f}] albedo[mean {a.mean():.3f} min {a.min():.2f}]',flush=True)
img10=np.array(mi.render(scene,sensor=0,integrator=integ,spp=128,seed=7))
print('after10 sensor0: mean %.4f (ref %.4f), PSNR %.2f'%(img10.mean(),r0.mean(),-10*np.log10(np.mean((img10-r0)**2))))
mi.Bitmap(img10).write(f'{ROOT}/inverse_results/diag_after10_s0.exr')
