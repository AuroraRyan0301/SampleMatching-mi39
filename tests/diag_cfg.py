import sys, drjit as dr, mitsuba as mi
mi.set_variant('cuda_ad_rgb')
sys.path.insert(0,'/path/to/SMmi39/sm_integrator_tests')
import stress_sm_lib as S
BAD=[({'probes_per_segment':0},'probes_per_segment'),
     ({'defer_capacity':0},'defer_capacity'),
     ({'segment_reservoir':8},'segment_reservoir'),
     ({'segment_reservoir':4,'linear_cost':True},'mutually exclusive')]
for extra,frag in BAD:
    try:
        mi.load_dict({**S.BASE,**extra}); print(f'[cfg] MISSED: {extra}',flush=True)
    except Exception as e:
        ok=frag in str(e)
        print(f'[cfg] {"OK  " if ok else "BAD "} {extra} -> {str(e)[:70]}',flush=True)
# warning path + good configs still load
for extra in ({},{'null_inner_loop':True},{'segment_reservoir':4},{'linear_cost':True}):
    mi.load_dict({**S.BASE,**extra})
print('[cfg] good configs load OK')
print('CFG DONE')
