"""Diagnose cam<->vehicle extrinsic convention in cameras.json by testing
which definition makes world_from_vehicle consistent across cameras.

Candidates per camera:
  A. cam_to_vehicle = inv( R,t from sensor_to_vehicle )           [current]
  B. cam_to_vehicle =      R,t from sensor_to_vehicle    (treat as cam->veh)
  C. cam_from_vehicle = R(cam_from_vehicle_qvec), t=cam_from_vehicle_tvec
     -> cam_to_vehicle = inv(that)
  D. cam_to_vehicle = R(cam_from_vehicle_qvec)^T ... (treat qvec/tvec as cam->veh)

For each candidate we compute world_from_vehicle via every camera for one
frame and report the max pairwise translation / rotation disagreement.
The correct convention yields ~0 disagreement (rigid rig).
"""
import json
import os
import sys
import numpy as np


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y],
    ], dtype=np.float64)


def load(data_path):
    with open(os.path.join(data_path, 'cameras.json')) as f:
        cj = json.load(f)
    entries = {}
    for e in cj:
        entries[e['camera']] = e
    # one frame's world_from_cam for each camera
    images_txt = os.path.join(data_path, 'sparse', 'txt', 'images.txt')
    frame = None
    wfc = {}
    with open(images_txt) as f:
        for line in f:
            if line.startswith('#'):
                continue
            p = line.split()
            if len(p) >= 10 and p[-1].endswith(('.jpg', '.png')):
                cam, fr = p[-1].split('/', 1)
                if frame is None:
                    frame = fr
                if fr != frame:
                    continue
                Rwc = qvec2rotmat(list(map(float, p[1:5])))
                cfw = np.eye(4); cfw[:3, :3] = Rwc
                cfw[:3, 3] = list(map(float, p[5:8]))
                wfc[cam] = np.linalg.inv(cfw)  # world_from_cam
    return entries, wfc, frame


def cam_to_vehicle_candidates(e):
    out = {}
    # A: inv(sensor_to_vehicle as given)
    r = e['sensor_to_vehicle']['rotation']; t = e['sensor_to_vehicle']['translation']
    M = np.eye(4); M[:3, :3] = qvec2rotmat([r['w'], r['x'], r['y'], r['z']])
    M[:3, 3] = [t['x'], t['y'], t['z']]
    out['A_inv_s2v'] = np.linalg.inv(M)
    # B: sensor_to_vehicle treated directly as cam->vehicle
    out['B_s2v_direct'] = M
    # C: cam_from_vehicle_qvec/tvec -> invert
    q = e['cam_from_vehicle_qvec']; t2 = e['cam_from_vehicle_tvec']
    Cfv = np.eye(4); Cfv[:3, :3] = qvec2rotmat(q); Cfv[:3, 3] = t2
    out['C_inv_cfv'] = np.linalg.inv(Cfv)
    # D: cam_from_vehicle_qvec/tvec treated directly as cam->vehicle
    out['D_cfv_direct'] = Cfv
    return out


def disagreement(wfv_list):
    ref = wfv_list[0]
    mr = mt = 0.0
    for M in wfv_list[1:]:
        dR = ref[:3, :3].T @ M[:3, :3]
        ang = np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
        dt = np.linalg.norm(ref[:3, 3] - M[:3, 3])
        mr = max(mr, ang); mt = max(mt, dt)
    return mr, mt


if __name__ == '__main__':
    dp = sys.argv[1] if len(sys.argv) > 1 else \
        '/media/zhangrujia/Call17812007125/3DGS/Datasets/colmap_test_full'
    entries, wfc, frame = load(dp)
    print('frame:', frame, '| cameras:', list(wfc.keys()))
    cand_names = ['A_inv_s2v', 'B_s2v_direct', 'C_inv_cfv', 'D_cfv_direct']
    for cn in cand_names:
        wfv = []
        for cam in wfc:
            c2v = cam_to_vehicle_candidates(entries[cam])[cn]
            wfv.append(wfc[cam] @ c2v)
        mr, mt = disagreement(wfv)
        print(f'  {cn:14s}: max d_rot={mr:7.3f} deg  max d_trans={mt:7.3f} m')
