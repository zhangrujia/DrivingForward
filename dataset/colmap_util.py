"""
Utilities for reading a COLMAP reconstruction (text format) for the
feed-forward evaluation of custom (self-collected) surround-view data.

Only what the DrivingForward eval pipeline needs is parsed:
  - per-camera intrinsics (PINHOLE: fx, fy, cx, cy) from cameras.json
  - per-camera sensor->vehicle extrinsics from cameras.json
  - per-image global pose (world<-cam) from sparse/txt/images.txt, used to
    derive the relative pose between a frame and its temporal neighbour.

The large per-image 2D-point lines in images.txt are skipped (only the pose
line of each image pair is parsed), so this stays fast and low-memory.
"""
import json
import os

import numpy as np


def qvec2rotmat(qvec):
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix (COLMAP convention)."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def read_cameras_json(path):
    """Read cameras.json produced alongside the COLMAP rig.

    Returns a dict keyed by camera name:
        {
          name: {
            'width', 'height',
            'K': 3x3 intrinsics (float32),
            'cam_to_vehicle': 4x4 (sensor->vehicle, i.e. nuScenes 'extrinsics')
          }
        }
    """
    with open(path, 'r') as f:
        data = json.load(f)

    cams = {}
    for entry in data:
        name = entry['camera']
        fx, fy, cx, cy = entry['params'][:4]
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        s2v = entry['sensor_to_vehicle']
        r = s2v['rotation']
        t = s2v['translation']
        R = qvec2rotmat([r['w'], r['x'], r['y'], r['z']])
        cam_to_vehicle = np.eye(4, dtype=np.float32)
        cam_to_vehicle[:3, :3] = R
        cam_to_vehicle[:3, 3] = [t['x'], t['y'], t['z']]

        cams[name] = {
            'width': int(entry['width']),
            'height': int(entry['height']),
            'K': K,
            'cam_to_vehicle': cam_to_vehicle,
        }
    return cams


def read_images_poses(path):
    """Parse sparse/txt/images.txt, returning world<-cam 4x4 per image name.

    COLMAP images.txt stores, on the first line of each image pair:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    where (Q, T) define the transform world->cam (cam_from_world):
        X_cam = R(Q) * X_world + T
    We return world_from_cam (cam->world) 4x4, i.e. the camera pose in world.

    The second line of each pair (2D points) is skipped.

    Returns: dict {image_name: 4x4 float64 cam->world}
    """
    poses = {}
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            # pose lines end with the image NAME (*.jpg / *.png)
            if len(parts) >= 10 and (parts[-1].endswith('.jpg')
                                     or parts[-1].endswith('.png')):
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                name = parts[-1]

                R_wc = qvec2rotmat([qw, qx, qy, qz])  # world->cam
                t_wc = np.array([tx, ty, tz], dtype=np.float64)

                cam_from_world = np.eye(4, dtype=np.float64)
                cam_from_world[:3, :3] = R_wc
                cam_from_world[:3, 3] = t_wc
                # invert to get cam->world (camera pose)
                world_from_cam = np.linalg.inv(cam_from_world)
                poses[name] = world_from_cam
    return poses


def list_frames_for_camera(poses, camera_name):
    """Return sorted list of frame filenames (e.g. '0000_...jpg') that have a
    pose for the given camera prefix. Sorted by the leading index so temporal
    neighbours are adjacent."""
    frames = []
    prefix = camera_name + '/'
    for name in poses.keys():
        if name.startswith(prefix):
            frames.append(name[len(prefix):])
    frames.sort()  # names start with zero-padded index -> temporal order
    return frames
