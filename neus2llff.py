import argparse
import os

import cv2
import numpy as np
from PIL import Image

from datasets.colmap_utils import rotmat2qvec, qvec2rotmat


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str,
                    default='./BlendedMVS_preprocessed/5a7d3db14989e929563eb153',
                    help='identify neus preprocesses root dir')
parser.add_argument('--exp_name', type=str, default='exp',
                    help='experiment name')
args = parser.parse_args()
root_dir = args.root_dir

camera_dict = np.load(os.path.join(root_dir, 'cameras_sphere.npz'))
n_images = len(camera_dict) // 6
world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

camera_intrinsics = []
camera_extrinsics = []
for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
    P = world_mat
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)
    camera_intrinsics.append(intrinsics[:3, :3])
    camera_extrinsics.append(pose)

img_dir = os.listdir(os.path.join(root_dir, 'image'))
sample_img = Image.open(os.path.join(root_dir, 'image', img_dir[0]))
W = sample_img.width
H = sample_img.height

os.makedirs('neus2llff_preprocessed', exist_ok=True)
root_dir = os.path.join('neus2llff_preprocessed', args.exp_name)
os.makedirs(root_dir, exist_ok=True)
os.makedirs(os.path.join(root_dir, 'sparse'), exist_ok=True)
os.makedirs(os.path.join(root_dir, 'sparse', '0'), exist_ok=True)
root_dir = os.path.join(root_dir, 'sparse', '0')

with open(os.path.join(root_dir, 'cameras.txt'), 'w') as f:
    # print(camera_intrinsics[0])
    for i in range(n_images):
        f.write(f'{i + 1} SIMPLE_PINHOLE {W} {H} {camera_intrinsics[i][0, 0]} {W // 2} {H // 2}\n')

with open(os.path.join(root_dir, 'images.txt'), 'w') as f:
    print(camera_extrinsics[0])
    # print(qvec2rotmat([0.83954509, 0.05650705, -0.43843359, -0.31582745]))
    for i in range(n_images):
        w2c_mat = camera_extrinsics[i]
        c2w_mat = np.linalg.inv(w2c_mat)
        qvec = rotmat2qvec(c2w_mat[:3, :3]).flat
        tvec = c2w_mat[:3, 3].flat
        f.write(f'{i + 1} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} {i + 1} {img_dir[0]}\n\n')

with open(os.path.join(root_dir, 'points3D.txt'), 'w') as f:
    pass