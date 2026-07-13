"""Which gradient pathway differs in cpp mode: probe-trans / probe-scat /
probes-off (leaves vertex-NEE attached + suffix). N=32 means."""
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
p=mi.traverse(scene); K='medium1.sigma_t.data'
sig=np.array(p[K]).ravel().astype(np.float64)
B={'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True}
import mitsuba.ad.integrators.prbvolpath_sm as smmod
cls=None
MODE={'m':'full'}
def make_patched(orig):
    def patched(self, scene_, medium, channel, alt_sampler, mei, seg_origin, seg_dir,
                interval, adj_trans, adj_scatt, nee_dir_sample, suffix_depth, active,
                include_indirect=True):
        if MODE['m']=='trans': adj_scatt=mi.Spectrum(0.0)
        elif MODE['m']=='scat': adj_trans=mi.Spectrum(0.0)
        elif MODE['m']=='off': adj_trans=mi.Spectrum(0.0); adj_scatt=mi.Spectrum(0.0)
        return orig(self, scene_, medium, channel, alt_sampler, mei, seg_origin, seg_dir,
                    interval, adj_trans, adj_scatt, nee_dir_sample, suffix_depth, active,
                    include_indirect)
    return patched
for name,extra in [('base',{}),('cpp',{'real_interaction_cpp':True})]:
    integ=mi.load_dict({**B,**extra}); cls=type(integ)
    orig=cls._sample_segment_probes
    cls._sample_segment_probes=make_patched(orig)
    for m in ('trans','scat','off'):
        MODE['m']=m; acc=None; N=32
        for it in range(N):
            dr.enable_grad(p[K]); p.update()
            img=mi.render(scene,p,sensor=1,integrator=integ,spp=8,spp_grad=8,seed=it,seed_grad=100+it)
            dr.backward(dr.sum(img,axis=None))
            g=np.array(dr.grad(p[K])).ravel().astype(np.float64)
            acc=g if acc is None else acc+g
            dr.disable_grad(p[K])
        print(f'[{name} {m}] N32<g,sig>={float(((acc/N)*sig).sum()):+.5e}',flush=True)
    cls._sample_segment_probes=orig
print('DIAG3 DONE')
