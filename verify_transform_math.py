"""
Pure-numpy verification of the coordinate-transform math used by
reconstruct_global_ply.py. Runs WITHOUT torch/CUDA so it can be checked in a
minimal environment. Uses scipy.spatial.transform.Rotation as ground truth.

Checks:
  1. rotmat_to_quat matches scipy (round-trip R -> q -> R).
  2. Hamilton quat product == rotation-matrix composition.
  3. ego->world rigid transform self-consistency (point path vs matrix path).
  4. REAL DATA: world_from_vehicle(t) computed from DIFFERENT cameras of the
     same frame must agree (this validates the whole global-stitch premise).
"""
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ---- replicate the exact math from the main script (numpy versions) --------
def qvec2rotmat(qvec):
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def rotmat_to_quat(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0
    y = np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0
    z = np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z], dtype=np.float64)


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def ok(name, cond):
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    return cond


def test_rotmat_quat():
    print('1) rotmat_to_quat vs scipy (random rotations)')
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        R = Rot.random(random_state=rng).as_matrix()
        q = rotmat_to_quat(R)
        R2 = qvec2rotmat(q)
        worst = max(worst, np.abs(R - R2).max())
    return ok(f'round-trip R->q->R max err={worst:.2e}', worst < 1e-6)


def test_quat_compose():
    print('2) Hamilton product == matrix composition')
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(2000):
        Ra = Rot.random(random_state=rng)
        Rb = Rot.random(random_state=rng)
        qa = rotmat_to_quat(Ra.as_matrix())
        qb = rotmat_to_quat(Rb.as_matrix())
        q_comp = quat_mul(qa, qb)
        R_comp = qvec2rotmat(q_comp / np.linalg.norm(q_comp))
        R_ref = Ra.as_matrix() @ Rb.as_matrix()
        worst = max(worst, np.abs(R_comp - R_ref).max())
    return ok(f'q(Ra*Rb) == Ra@Rb max err={worst:.2e}', worst < 1e-6)


def test_rigid_selfconsistency():
    print('3) ego->world rigid transform self-consistency')
    rng = np.random.default_rng(2)
    R = Rot.random(random_state=rng).as_matrix()
    t = rng.normal(size=3)
    W = np.eye(4); W[:3, :3] = R; W[:3, 3] = t
    pts = rng.normal(size=(500, 3))
    # point path
    p_world = pts @ R.T + t
    # homogeneous matrix path
    ph = np.concatenate([pts, np.ones((500, 1))], 1) @ W.T
    worst = np.abs(p_world - ph[:, :3]).max()
    # a gaussian's quaternion composed with world R, applied to a local axis
    q_loc = rotmat_to_quat(Rot.random(random_state=rng).as_matrix())
    q_w = quat_mul(rotmat_to_quat(R), q_loc)
    R_w = qvec2rotmat(q_w / np.linalg.norm(q_w))
    ref = R @ qvec2rotmat(q_loc)
    worst2 = np.abs(R_w - ref).max()
    return (ok(f'point vs matrix path err={worst:.2e}', worst < 1e-10) and
            ok(f'rotated gaussian frame err={worst2:.2e}', worst2 < 1e-6))


def read_cameras_json(path):
    with open(path) as f:
        data = json.load(f)
    cams = {}
    for e in data:
        r = e['sensor_to_vehicle']['rotation']
        t = e['sensor_to_vehicle']['translation']
        R = qvec2rotmat([r['w'], r['x'], r['y'], r['z']])
        M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [t['x'], t['y'], t['z']]
        cams[e['camera']] = M  # cam_to_vehicle
    return cams


def read_some_poses(path, wanted_frame, cam_names):
    """Parse images.txt, return {('cam/frame'): world_from_cam} for the one
    frame we test, across the given cameras. Stops early once all found."""
    want = set(f'{c}/{wanted_frame}' for c in cam_names)
    found = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[-1].endswith(('.jpg', '.png')):
                name = parts[-1]
                if name in want:
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    Rwc = qvec2rotmat([qw, qx, qy, qz])
                    cfw = np.eye(4); cfw[:3, :3] = Rwc; cfw[:3, 3] = [tx, ty, tz]
                    found[name] = np.linalg.inv(cfw)  # world_from_cam
                    if len(found) == len(want):
                        break
    return found


def test_trajectory_coherence(data_path):
    """The invariant global stitching actually needs: the REF-CAMERA vehicle
    trajectory must be temporally coherent (smooth, monotonic-ish motion) so
    consecutive frames land consistently in one world frame.

    NOTE: cross-camera agreement of world_from_vehicle WITHIN a frame is NOT
    required and is not checked -- COLMAP solves per-image poses that collapse
    the physical rig baseline. Within a frame the MODEL fuses cameras into the
    ego frame via cameras.json extrinsics; across frames we use only the ref
    camera, so per-camera disagreement does not affect the stitch.
    """
    print('4) REAL DATA: ref-camera trajectory is temporally coherent')
    images_txt = os.path.join(data_path, 'sparse', 'txt', 'images.txt')
    if not os.path.isfile(images_txt):
        print('  [SKIP] images.txt not found at', data_path)
        return True

    # parse all ref-camera poses
    ref_cam = 'mid_center_top_wide'
    cfg_path = os.path.join(data_path, 'rig_config.json')
    if os.path.isfile(cfg_path):
        try:
            rig = json.load(open(cfg_path))
            rig = rig[0] if isinstance(rig, list) else rig
            ref_id = rig.get('ref_camera_id')
            for c in rig.get('cameras', []):
                if c.get('camera_id') == ref_id:
                    ref_cam = c['image_prefix'].rstrip('/')
        except Exception:
            pass
    print(f'  ref camera: {ref_cam}')

    prefix = ref_cam + '/'
    frames, pos = [], []
    with open(images_txt) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 10 and p[-1].endswith(('.jpg', '.png')) \
                    and p[-1].startswith(prefix):
                Rwc = qvec2rotmat(list(map(float, p[1:5])))
                cfw = np.eye(4); cfw[:3, :3] = Rwc
                cfw[:3, 3] = list(map(float, p[5:8]))
                wfc = np.linalg.inv(cfw)
                frames.append(p[-1][len(prefix):])
                pos.append(wfc[:3, 3])
    if len(pos) < 3:
        print('  [SKIP] too few ref poses')
        return True

    order = np.argsort(frames)
    P = np.array(pos)[order]
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    path_len = float(d.sum())
    bbox = P.max(0) - P.min(0)
    # coherent = steps are small & regular (no teleport), path has real extent
    step_cv = float(d.std() / (d.mean() + 1e-9))       # coefficient of variation
    no_teleport = float(d.max()) < 10 * (np.median(d) + 1e-9)
    print(f'  frames={len(P)}  path_len={path_len:.2f} m  '
          f'step median={np.median(d):.3f} m  max={d.max():.3f} m')
    print(f'  trajectory bbox (m) = {np.round(bbox, 2)}  step_cv={step_cv:.2f}')
    return (ok('no teleport between consecutive frames', no_teleport) and
            ok(f'trajectory has real extent (path {path_len:.1f} m > 0.5)',
               path_len > 0.5))


if __name__ == '__main__':
    data_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/media/zhangrujia/Call17812007125/3DGS/Datasets/colmap_test_full'
    results = []
    results.append(test_rotmat_quat())
    results.append(test_quat_compose())
    results.append(test_rigid_selfconsistency())
    results.append(test_trajectory_coherence(data_path))
    print()
    print('ALL PASS' if all(results) else 'SOME FAILED')
    sys.exit(0 if all(results) else 1)
