"""Overnight stress campaign for prbvolpath_sm (fused C++ walk + deferred
probes, the PR configuration). Usage: stress_sm.py <group> [variant]

Groups: boundary | emitter | media | depthrr | lightsgrad | camera | speed
Each cell prints one structured line:
  CELL <group>/<name> <PASS|FAIL|WARN|INFO> key=value ...
and appends JSON to stress_results/<group>_<variant>.jsonl
"""
import sys, json, time, os
import numpy as np
import drjit as dr
import mitsuba as mi

GROUP = sys.argv[1]
VARIANT = sys.argv[2] if len(sys.argv) > 2 else 'cuda_ad_rgb'
mi.set_variant(VARIANT)
dr.set_thread_count(16)

ROOT = '/path/to/SMmi39'
OUT = f'{ROOT}/stress_results/{GROUP}_{VARIANT}.jsonl'
T = mi.ScalarTransform4f

SM = {'quad': {}, 'lin': {'linear_cost': True}, 'k4': {'segment_reservoir': 4}}
BASE = {'type': 'prbvolpath_sm', 'max_depth': 32, 'rr_depth': 1032,
        'probes_per_segment': 4}

def emit(name, status, **kw):
    rec = {'cell': f'{GROUP}/{name}', 'status': status, **kw}
    print('CELL', rec['cell'], status,
          ' '.join(f'{k}={v}' for k, v in kw.items()), flush=True)
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')

def sigma_grid():
    return np.load(f'{ROOT}/data/stress/sigma_blob.npy')

def albedo_ramp():
    return np.load(f'{ROOT}/data/stress/albedo_ramp.npy')

def make_scene(boundary='box', emitter='constant', albedo='ramp',
               homogeneous=False, supergrid=8, surface=False,
               phase_g=None, res=64, spp=8):
    d = {'type': 'scene',
         'integrator': {'type': 'volpath'},
         'sensor': {'type': 'perspective', 'fov': 45,
                    'to_world': T().look_at([0, 0, 4.2], [0, 0, 0], [0, 1, 0]),
                    'film': {'type': 'hdrfilm', 'width': res, 'height': res,
                             'rfilter': {'type': 'box'}, 'pixel_format': 'rgb'},
                    'sampler': {'type': 'independent', 'sample_count': spp}}}
    # ---- lighting
    if emitter == 'constant':
        d['light'] = {'type': 'constant', 'radiance': 1.0}
    elif emitter == 'envmap':
        d['light'] = {'type': 'envmap',
                      'filename': f'{ROOT}/data/scenes/common/uniform_white.exr',
                      'scale': 1.0}
    elif emitter == 'area_out':
        d['light'] = {'type': 'rectangle',
                      'to_world': T().translate([0, 2.5, 0]).rotate([1, 0, 0], 90).scale(1.2),
                      'emitter': {'type': 'area', 'radiance': 6.0}}
    elif emitter == 'area_in':
        d['light'] = {'type': 'rectangle',
                      'to_world': T().translate([0, 0, -0.5]).scale(0.25),
                      'emitter': {'type': 'area', 'radiance': 25.0}}
        d['fill'] = {'type': 'constant', 'radiance': 0.05}
    elif emitter == 'point':
        d['light'] = {'type': 'point', 'position': [2.5, 2.5, 2.5],
                      'intensity': 40.0}
        d['fill'] = {'type': 'constant', 'radiance': 0.02}
    # ---- medium
    if homogeneous:
        interior = {'type': 'homogeneous', 'sigma_t': 1.2, 'albedo': 0.8}
    else:
        interior = {'type': 'heterogeneous',
                    'sigma_t': {'type': 'gridvolume',
                                'grid': mi.VolumeGrid(sigma_grid()),
                                'to_world': T().translate([-1, -1, -1]).scale(2.0)},
                    'albedo': ({'type': 'gridvolume',
                                'grid': mi.VolumeGrid(albedo_ramp()),
                                'to_world': T().translate([-1, -1, -1]).scale(2.0)}
                               if albedo == 'ramp' else 0.8),
                    'scale': 2.0,
                    'majorant_resolution_factor': supergrid,
                    'majorant_factor': 1.3,
                    'sample_emitters': True}
    if phase_g is not None:
        interior['phase'] = {'type': 'hg', 'g': phase_g}
    shape = {'bsdf': {'type': 'null'}, 'interior': interior}
    if boundary == 'box':
        shape.update({'type': 'cube'})
    elif boundary == 'sphere':
        shape.update({'type': 'sphere', 'radius': 1.15})
    elif boundary == 'torus':
        shape.update({'type': 'obj',
                      'filename': f'{ROOT}/data/stress/torus.obj',
                      'to_world': T().scale(1.4)})
    d['medium_shape'] = shape
    if surface:
        d['blocker'] = {'type': 'sphere', 'radius': 0.25,
                        'to_world': T().translate([0.4, 0.0, 0.0]),
                        'bsdf': {'type': 'diffuse',
                                 'reflectance': {'type': 'rgb',
                                                 'value': [0.7, 0.3, 0.2]}}}
    return mi.load_dict(d)

# --------------------------------------------------------------------------
def render_sum(scene, integ, spp, seed):
    img = mi.render(scene, sensor=0, integrator=integ, spp=spp, seed=seed)
    return float(np.array(img, copy=False).sum())

def primal_parity(scene, sm_dict, n=6, spp=256):
    ref = mi.load_dict({'type': 'volpath', 'max_depth': 32, 'rr_depth': 1032})
    it = mi.load_dict(sm_dict)
    a = np.array([render_sum(scene, ref, spp, 10 + s) for s in range(n)])
    b = np.array([render_sum(scene, it, spp, 10 + s) for s in range(n)])
    se = np.sqrt(a.var() / n + b.var() / n)
    z = abs(a.mean() - b.mean()) / max(se, 1e-9)
    rel = abs(a.mean() - b.mean()) / max(abs(a.mean()), 1e-9)
    return rel, z

def grad_param(scene, sm_dict, key, direction, n=48, spp=8,
               fd_eps=2e-2, fd_spp=2048, fd_int=None, fd_forward=False):
    """Adjoint <g, direction> (N-seed mean +- SE) vs FD along `direction`
    with correlated seeds. fd_forward=True uses the one-sided difference
    [I(base + eps*dir) - I(base)] / eps — REQUIRED whenever `direction` is
    nonzero at voxels where the parameter is 0: a central difference would
    need negative sigma there, which the renderer clamps, silently halving
    those voxels' contribution."""
    p = mi.traverse(scene)
    base = np.array(p[key], copy=True)
    integ = mi.load_dict(sm_dict)
    acc = []
    for s in range(n):
        dr.enable_grad(p[key]); p.update()
        img = mi.render(scene, p, sensor=0, integrator=integ,
                        spp=spp, spp_grad=spp, seed=s, seed_grad=500 + s)
        dr.backward(dr.sum(img, axis=None))
        g = np.array(dr.grad(p[key]), copy=False).ravel().astype(np.float64)
        dr.disable_grad(p[key])
        g = np.where(np.isfinite(g), g, 0)
        acc.append(float((g * direction.ravel()).sum()))
    acc = np.array(acc)
    ad, ad_se = acc.mean(), acc.std() / np.sqrt(n)
    # FD (primal only, unbiased referee)
    fd_int = fd_int or mi.load_dict({'type': 'volpath', 'max_depth': 32,
                                     'rr_depth': 1032})
    fds = []
    for s in (7, 8):
        hi = base + fd_eps * direction.reshape(base.shape)
        if fd_forward:
            lo, h2 = base, fd_eps
        else:
            lo = base - fd_eps * direction.reshape(base.shape)
            h2 = 2 * fd_eps
        p[key] = mi.TensorXf(hi.astype(np.float32)) if base.ndim > 1 else type(p[key])(hi)
        p.update()
        Lp = render_sum(scene, fd_int, fd_spp, s)
        p[key] = mi.TensorXf(lo.astype(np.float32)) if base.ndim > 1 else type(p[key])(lo)
        p.update()
        Lm = render_sum(scene, fd_int, fd_spp, s)
        fds.append((Lp - Lm) / h2)
    p[key] = mi.TensorXf(base.astype(np.float32)) if base.ndim > 1 else type(p[key])(base)
    p.update()
    fd = float(np.mean(fds))
    return ad, ad_se, fd, fds

def check_grad(scene, name, sm_dict, key_sub, control=False):
    p = mi.traverse(scene)
    keys = [k for k in p.keys() if key_sub in k]
    if not keys:
        emit(name, 'WARN', note=f'no parameter matching {key_sub}')
        return False
    key = keys[0]
    base = np.array(p[key], copy=True)
    dirn = base.copy()  # direction = parameter value: <g, theta>
    ad, se, fd, fds = grad_param(scene, sm_dict, key, dirn)
    tol = 3 * se + 0.06 * abs(fd) + 0.5 * abs(fds[0] - fds[1])
    ok = abs(ad - fd) <= tol
    st = 'PASS' if ok else ('WARN' if control else 'FAIL')
    emit(name, st, ad=f'{ad:.4e}', se=f'{se:.1e}',
         fd=f'{fd:.4e}', ratio=f'{ad/fd if fd else float("nan"):.3f}')
    return ok

def check_grad_sigma(scene, name, sm_dict):
    return check_grad(scene, name, sm_dict, 'sigma_t.data')

# ============================ groups ============================
if GROUP == 'boundary':
    for b in ('box', 'sphere', 'torus'):
        scene = make_scene(boundary=b)
        for nm, extra in SM.items():
            rel, z = primal_parity(scene, {**BASE, **extra})
            emit(f'{b}_{nm}_primal', 'PASS' if z < 4 else 'FAIL',
                 rel=f'{rel:.2e}', z=f'{z:.1f}')
            check_grad_sigma(scene, f'{b}_{nm}_gsigma', {**BASE, **extra})

elif GROUP == 'emitter':
    for e in ('constant', 'envmap', 'area_out', 'area_in', 'point'):
        scene = make_scene(emitter=e)
        for nm, extra in SM.items():
            rel, z = primal_parity(scene, {**BASE, **extra})
            emit(f'{e}_{nm}_primal', 'PASS' if z < 4 else 'FAIL',
                 rel=f'{rel:.2e}', z=f'{z:.1f}')
            check_grad_sigma(scene, f'{e}_{nm}_gsigma', {**BASE, **extra})

elif GROUP == 'media':
    # homogeneous medium (cpp fallback path + nee_handle_homogeneous)
    scene = make_scene(homogeneous=True)
    for nm, extra in SM.items():
        rel, z = primal_parity(scene, {**BASE, **extra})
        emit(f'homog_{nm}_primal', 'PASS' if z < 4 else 'FAIL',
             rel=f'{rel:.2e}', z=f'{z:.1f}')
    # supergrid off (global majorant, fused falls back to naive walk)
    scene = make_scene(supergrid=0)
    for nm, extra in SM.items():
        rel, z = primal_parity(scene, {**BASE, **extra})
        emit(f'sg0_{nm}_primal', 'PASS' if z < 4 else 'FAIL',
             rel=f'{rel:.2e}', z=f'{z:.1f}')
        check_grad_sigma(scene, f'sg0_{nm}_gsigma', {**BASE, **extra})
    # ramp albedo: sigma gradient + ALBEDO-GRID gradient vs FD
    scene = make_scene(albedo='ramp')
    p = mi.traverse(scene)
    akey = [k for k in p.keys() if 'albedo.data' in k][0]
    aw = np.array(p[akey]).size
    dirn = np.ones(aw)  # additive uniform direction on albedo
    for nm, extra in SM.items():
        check_grad_sigma(scene, f'ramp_{nm}_gsigma', {**BASE, **extra})
        ad, se, fd, fds = grad_param(scene, {**BASE, **extra}, akey, dirn,
                                     fd_eps=5e-3)
        tol = 3 * se + 0.06 * abs(fd) + 0.5 * abs(fds[0] - fds[1])
        emit(f'ramp_{nm}_galbedo', 'PASS' if abs(ad - fd) <= tol else 'FAIL',
             ad=f'{ad:.4e}', se=f'{se:.1e}', fd=f'{fd:.4e}')
    # surface (mesh with real BSDF) inside medium
    scene = make_scene(surface=True)
    for nm, extra in SM.items():
        rel, z = primal_parity(scene, {**BASE, **extra})
        emit(f'surface_{nm}_primal', 'PASS' if z < 4 else 'FAIL',
             rel=f'{rel:.2e}', z=f'{z:.1f}')
        check_grad_sigma(scene, f'surface_{nm}_gsigma', {**BASE, **extra})

elif GROUP == 'depthrr':
    # deep scattering: dense sigma, high albedo; RR on and off
    for rr, md in (('rroff', 1032), ('rron', 8)):
        for nm, extra in SM.items():
            d = {**BASE, **extra, 'max_depth': 64, 'rr_depth': md if md != 1032 else 1064}
            scene = make_scene(albedo='const')
            p = mi.traverse(scene)
            skey = [k for k in p.keys() if 'sigma_t.data' in k][0]
            p[skey] = mi.TensorXf(np.full((8, 8, 8, 1), 2.5, np.float32))
            p.update()
            rel, z = primal_parity(scene, d)
            emit(f'deep_{rr}_{nm}_primal', 'PASS' if z < 4 else 'FAIL',
                 rel=f'{rel:.2e}', z=f'{z:.1f}')
            check_grad_sigma(scene, f'deep_{rr}_{nm}_gsigma', d)

elif GROUP == 'lightsgrad':
    # emitter RADIANCE (color) gradient: constant + area emitters
    for e, keyfrag in (('constant', 'radiance'), ('area_out', 'radiance')):
        scene = make_scene(emitter=e)
        p = mi.traverse(scene)
        ks = [k for k in p.keys() if keyfrag in k and 'value' in k]
        if not ks:
            ks = [k for k in p.keys() if keyfrag in k]
        if not ks:
            emit(f'{e}_gradiance', 'WARN', reason='no differentiable radiance param',
                 keys=';'.join(list(p.keys())[:8]))
            continue
        key = ks[0]
        w = np.array(p[key]).size
        dirn = np.ones(w)
        for nm, extra in SM.items():
            ad, se, fd, fds = grad_param(scene, {**BASE, **extra}, key, dirn,
                                         fd_eps=2e-2)
            tol = 3 * se + 0.06 * abs(fd) + 0.5 * abs(fds[0] - fds[1])
            emit(f'{e}_{nm}_gradiance', 'PASS' if abs(ad - fd) <= tol else 'FAIL',
                 key=key, ad=f'{ad:.4e}', fd=f'{fd:.4e}', se=f'{se:.1e}')
    # emitter POSITION gradient: documented limitation of detached sampling
    scene = make_scene(emitter='area_out')
    p = mi.traverse(scene)
    vks = [k for k in p.keys() if 'vertex_positions' in k and 'light' in k.lower()]
    if not vks:
        vks = [k for k in p.keys() if 'vertex_positions' in k]
    if vks:
        key = vks[0]
        base = np.array(p[key], copy=True)
        dirn = np.zeros_like(base); dirn[1::3] = 1.0  # translate along +y
        for nm, extra in list(SM.items())[:1]:
            try:
                ad, se, fd, fds = grad_param(scene, {**BASE, **extra}, key, dirn,
                                             fd_eps=2e-2)
                ok = abs(ad - fd) <= 3 * se + 0.1 * abs(fd) + abs(fds[0] - fds[1])
                emit(f'area_pos_{nm}', 'PASS' if ok else 'WARN',
                     note='geometric grads documented-unsupported (detached sampling)',
                     ad=f'{ad:.4e}', fd=f'{fd:.4e}', se=f'{se:.1e}')
            except Exception as ex:
                emit(f'area_pos_{nm}', 'WARN',
                     note='attached emitter geometry makes the upstream NEE '
                          'loop demand max_iterations; geometric grads are '
                          'documented-unsupported for the prbvolpath family',
                     error=str(ex)[:120])
    else:
        emit('area_pos', 'WARN', reason='no vertex_positions param')

elif GROUP == 'camera':
    scene = make_scene()
    p = mi.traverse(scene)
    cks = [k for k in p.keys() if 'to_world' in k and
           ('ensor' in k or 'amera' in k or 'erspective' in k)]
    emit('cam_param_discovery', 'INFO', keys=';'.join(cks) if cks else 'NONE',
         allkeys=';'.join([k for k in p.keys()][:20]))
    if cks:
        key = cks[0]
        for nm, extra in list(SM.items())[:1]:
            try:
                integ = mi.load_dict({**BASE, **extra})
                dr.enable_grad(p[key]); p.update()
                img = mi.render(scene, p, sensor=0, integrator=integ,
                                spp=8, spp_grad=8, seed=1, seed_grad=2)
                dr.backward(dr.sum(img, axis=None))
                g = dr.grad(p[key])
                gn = float(np.abs(np.array(g, copy=False)).sum())
                dr.disable_grad(p[key])
                emit(f'cam_pose_{nm}', 'INFO',
                     note='no crash; gradient magnitude reported (correctness of '
                          'sensor-pose grads not claimed by prbvolpath family)',
                     grad_abs_sum=f'{gn:.3e}')
            except Exception as ex:
                emit(f'cam_pose_{nm}', 'WARN', error=str(ex)[:160])

elif GROUP == 'speed':
    import itertools
    res, spp = 512, 8
    scene_cfgs = [('blob', dict()), ('dense', dict())]
    rows = []
    for scn, _ in scene_cfgs:
        scene = make_scene(res=res)
        if scn == 'dense':
            p = mi.traverse(scene)
            skey = [k for k in p.keys() if 'sigma_t.data' in k][0]
            p[skey] = mi.TensorXf(np.full((8, 8, 8, 1), 2.5, np.float32))
            p.update()
        p = mi.traverse(scene)
        skey = [k for k in p.keys() if 'sigma_t.data' in k][0]
        for md, rr in ((16, 1016), (64, 1064), (64, 8)):
            for nm, dct in [('prb', {'type': 'prbvolpath'})] + \
                           [(k, {**BASE, **v}) for k, v in SM.items()]:
                d = {**dct, 'max_depth': md, 'rr_depth': rr}
                integ = mi.load_dict(d)
                # primal
                ts = []
                for it in range(3):
                    t0 = time.time()
                    img = mi.render(scene, sensor=0, integrator=integ,
                                    spp=64, seed=it)
                    dr.eval(img); dr.sync_thread()
                    ts.append(time.time() - t0)
                tp = min(ts[1:])
                # adjoint
                ts = []
                for it in range(3):
                    t0 = time.time()
                    dr.enable_grad(p[skey]); p.update()
                    img = mi.render(scene, p, sensor=0, integrator=integ,
                                    spp=spp, spp_grad=spp, seed=it, seed_grad=99 + it)
                    dr.backward(dr.sum(img, axis=None))
                    dr.eval(dr.grad(p[skey])); dr.sync_thread()
                    dr.disable_grad(p[skey])
                    ts.append(time.time() - t0)
                ta = min(ts[1:])
                emit(f'speed_{scn}_d{md}_rr{"on" if rr < 100 else "off"}_{nm}',
                     'INFO', primal_s=f'{tp:.3f}', adjoint_s=f'{ta:.3f}')

elif GROUP == 'gradfull':
    # Full derivative sweep: every boundary x emitter combo, sigma AND albedo
    # gradients, quad/lin/k4 plus a prbvolpath AD control column (control
    # cells report WARN instead of FAIL: they arbitrate the FD referee).
    PRB = {'type': 'prbvolpath', 'max_depth': 32, 'rr_depth': 1032}
    b = os.environ.get('GF_BOUNDARY', 'box')
    for e in ('constant', 'envmap', 'area_out', 'area_in', 'point'):
        scene = make_scene(boundary=b, emitter=e)
        for nm, extra in SM.items():
            check_grad(scene, f'{b}_{e}_{nm}_gsigma', {**BASE, **extra},
                       'sigma_t.data')
            check_grad(scene, f'{b}_{e}_{nm}_galbedo', {**BASE, **extra},
                       'albedo.data')
        check_grad(scene, f'{b}_{e}_prb_gsigma', PRB, 'sigma_t.data',
                   control=True)
        check_grad(scene, f'{b}_{e}_prb_galbedo', PRB, 'albedo.data',
                   control=True)

elif GROUP == 'gphase':
    # Henyey-Greenstein g derivative; prbvolpath as the reference for what
    # the family supports.
    PRB = {'type': 'prbvolpath', 'max_depth': 32, 'rr_depth': 1032}
    for g0 in (0.5, -0.3):
        scene = make_scene(phase_g=g0)
        for nm, extra in SM.items():
            rel, z = primal_parity(scene, {**BASE, **extra})
            emit(f'hg{g0}_{nm}_primal', 'PASS' if z < 4 else 'FAIL',
                 rel=f'{rel:.2e}', z=f'{z:.1f}')
            check_grad(scene, f'hg{g0}_{nm}_gphase', {**BASE, **extra}, '.g')
        check_grad(scene, f'hg{g0}_prb_gphase', PRB, '.g', control=True)

elif GROUP == 'gones':
    # <g, 1> with EXACT-ZERO voxels in the grid, forward FD (see grad_param).
    # The all-ones direction weights empty voxels fully, unlike <g, sigma>
    # (zero weight exactly there) — this is the projection that caught the
    # empty-voxel scatter-gradient loss (commit bd6f228). prbvolpath is
    # emitted as a control: its detached estimator is known-biased in this
    # projection on dense-empty mixes (documented, WARN not FAIL).
    scene = make_scene()
    p = mi.traverse(scene)
    key = [k for k in p.keys() if 'sigma_t.data' in k][0]
    carved = np.array(p[key], copy=True)
    carved[carved < 0.6] = 0.0            # guarantee real empty voxels
    p[key] = mi.TensorXf(carved.astype(np.float32)); p.update()
    assert (carved == 0).mean() > 0.2, 'gones scene lost its empty voxels'
    ones = np.ones_like(carved)
    PRB = {'type': 'prbvolpath', 'max_depth': 32, 'rr_depth': 1032}
    for nm, d in [(k, {**BASE, **v}) for k, v in SM.items()] + [('prb', PRB)]:
        ad, se, fd, fds = grad_param(scene, d, key, ones,
                                     fd_eps=2e-3, fd_forward=True)
        tol = 3 * se + 0.06 * abs(fd) + 0.5 * abs(fds[0] - fds[1])
        ok = abs(ad - fd) <= tol
        st = 'PASS' if ok else ('WARN' if nm == 'prb' else 'FAIL')
        emit(f'zeros_{nm}_gones', st, ad=f'{ad:.4e}', se=f'{se:.1e}',
             fd=f'{fd:.4e}', ratio=f'{ad/fd if fd else float("nan"):.3f}')

print('GROUP DONE', GROUP, VARIANT, flush=True)
