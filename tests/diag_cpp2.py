"""Compare probe-call statistics (interval sums, seg counts, adj magnitudes)
between baseline and cpp mode."""
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
B={'type':'prbvolpath_sm','max_depth':64,'rr_depth':1064,'probes_per_segment':4,'linear_cost':True}
STATS={}
import mitsuba.ad.integrators.prbvolpath_sm as smmod
for name,extra in [('base',{}),('cpp',{'real_interaction_cpp':True})]:
    integ=mi.load_dict({**B,**extra})
    cls=type(integ)
    orig=cls.__dict__.get('_sample_segment_probes_orig') or cls._sample_segment_probes
    st={'n':0.0,'isum':0.0,'imax':0.0,'atsum':0.0,'assum':0.0}
    def patched(self, scene_, medium, channel, alt_sampler, mei, seg_origin, seg_dir,
                interval, adj_trans, adj_scatt, nee_dir_sample, suffix_depth, active,
                include_indirect=True, _st=st, _orig=orig):
        a=mi.Float(dr.select(active, 1.0, 0.0))
        _st['n']+=float(dr.sum(a)[0]); _st['isum']+=float(dr.sum(interval*a)[0])
        _st['imax']=max(_st['imax'], float(dr.max(dr.select(active,interval,0.0))[0]))
        _st['atsum']+=float(dr.sum(dr.mean(dr.detach(adj_trans))*a)[0])
        _st['assum']+=float(dr.sum(dr.mean(dr.detach(adj_scatt))*a)[0])
        return _orig(self, scene_, medium, channel, alt_sampler, mei, seg_origin, seg_dir,
                     interval, adj_trans, adj_scatt, nee_dir_sample, suffix_depth, active,
                     include_indirect)
    cls._sample_segment_probes = patched
    dr.enable_grad(p[K]); p.update()
    img=mi.render(scene,p,sensor=1,integrator=integ,spp=4,spp_grad=4,seed=3,seed_grad=103)
    dr.backward(dr.sum(img,axis=None))
    dr.eval(dr.grad(p[K])); dr.disable_grad(p[K])
    cls._sample_segment_probes = orig
    print(f"[{name}] segs={st['n']:.3e} mean_interval={st['isum']/max(st['n'],1):.4f} "
          f"max_interval={st['imax']:.3f} mean|adj_t|={st['atsum']/max(st['n'],1):.4f} "
          f"mean|adj_s|={st['assum']/max(st['n'],1):.4f}",flush=True)
print('DIAG2 DONE')
