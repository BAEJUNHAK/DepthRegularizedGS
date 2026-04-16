"""
calib_XXXX.ini + depth_raw_XXXX.png → COLMAP binary format 변환기.

저자의 원래 파이프라인(COLMAP → ZoeDepth → optimize_depth)을 그대로 사용하기 위해
우리 데이터를 COLMAP이 출력한 것처럼 변환합니다.

출력 구조:
    <output_dir>/
    ├── images/
    │   ├── 00000.png
    │   └── ...
    ├── sparse/
    │   └── 0/
    │       ├── cameras.bin
    │       ├── images.bin
    │       └── points3D.bin
    └── split_index.json

Usage:
    python scripts/convert_custom_to_colmap.py \
        --input_dir data/mitsubishi_raw \
        --output_dir data/mitsubishi \
        --train_ratio 0.8 \
        --split_seed 42 \
        --depth_subsample 50
"""

import os
import sys
import struct
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copy2
import cv2

# ─── calib INI 파싱 ───

def parse_calib_ini(filepath):
    params = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                params[key.strip()] = value.strip()

    def parse_matrix(s):
        parts = s.split(':')
        rows, cols = int(parts[1]), int(parts[2])
        vals = [float(x) for x in parts[3].split(',')]
        return np.array(vals).reshape(rows, cols)

    def parse_vector(s):
        parts = s.split(':')
        vals = [float(x) for x in parts[2].split(',')]
        return np.array(vals)

    K = parse_matrix(params['k_matrix'])
    R = parse_matrix(params['r_matrix'])
    T = parse_vector(params['t_vector'])
    width = int(params['width'])
    height = int(params['height'])
    return K, R, T, width, height


# ─── rotation matrix → quaternion (COLMAP: w, x, y, z) ───

def rotmat2qvec(R):
    """Rotation matrix to COLMAP quaternion (w, x, y, z)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# ─── GT depth → sparse 3D points (COLMAP 시뮬레이션) ───

def create_sparse_points_from_depth(calib_data, depth_dir, n_views,
                                     subsample_pixels=50, min_views=2):
    """
    GT depth에서 sparse 3D points를 추출합니다.
    COLMAP이 만든 것처럼 여러 뷰에서 관측된 points만 남깁니다.

    subsample_pixels: 각 뷰에서 몇 픽셀마다 1개씩 샘플링 (50=50px 간격 grid)
    min_views: 최소 몇 뷰에서 관측되어야 유효한 point인지
    """
    all_points_3d = {}  # point_id -> {xyz, rgb, error, seen_by: [(img_id, xy)]}
    point_id_counter = 1

    # 각 뷰에서 sparse depth sampling → 3D point 생성
    for idx in tqdm(range(n_views), desc="Generating sparse points"):
        K, R, T, width, height = calib_data[idx]

        depth_path = os.path.join(depth_dir, f"depth_raw_{idx:04d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth_raw.astype(np.float64) * 0.01  # meters

        rgb_path = os.path.join(os.path.dirname(depth_dir), f"rgb_{idx:04d}.png")
        if os.path.exists(rgb_path):
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = np.ones((height, width, 3), dtype=np.uint8) * 128

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # subsample grid
        ys = np.arange(0, height, subsample_pixels)
        xs = np.arange(0, width, subsample_pixels)
        uu, vv = np.meshgrid(xs, ys)
        uu, vv = uu.ravel(), vv.ravel()

        dd = depth[vv, uu]
        mask = dd > 0
        uu, vv, dd = uu[mask], vv[mask], dd[mask]

        # back-project to 3D
        x_cam = (uu - cx) * dd / fx
        y_cam = (vv - cy) * dd / fy
        z_cam = dd
        p_cam = np.stack([x_cam, y_cam, z_cam], axis=1)

        # camera → world
        # w2c: p_cam = R @ p_world + T
        # p_world = R^T @ (p_cam - T)
        p_world = (R.T @ (p_cam.T - T.reshape(3, 1))).T

        colors = rgb_img[vv, uu]

        for j in range(len(p_world)):
            pid = point_id_counter
            point_id_counter += 1
            all_points_3d[pid] = {
                'xyz': p_world[j],
                'rgb': colors[j],
                'error': 1.0,
                'observations': [(idx + 1, uu[j].astype(float), vv[j].astype(float))]
            }

    # 다른 뷰에서도 관측 가능한지 체크하여 multi-view points 만들기
    # (진짜 COLMAP처럼 feature matching 대신, reprojection으로 시뮬레이션)
    print("Checking multi-view visibility...")
    final_points = {}
    point_id_new = 1

    # 모든 points를 numpy로 변환
    all_pids = list(all_points_3d.keys())
    all_xyz = np.array([all_points_3d[pid]['xyz'] for pid in all_pids])

    for cam_idx in tqdm(range(n_views), desc="Reprojecting"):
        K, R, T, width, height = calib_data[cam_idx]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # project all points to this camera
        p_cam = (R @ all_xyz.T + T.reshape(3, 1))  # (3, N)
        z = p_cam[2]
        u_proj = (p_cam[0] / z) * 1.0  # already in pixel coords via K
        # Actually need to apply K
        u_proj = fx * p_cam[0] / z + cx
        v_proj = fy * p_cam[1] / z + cy

        # valid: in front of camera and within image bounds
        valid = (z > 0) & (u_proj >= 0) & (u_proj < width) & (v_proj >= 0) & (v_proj < height)

        for j in np.where(valid)[0]:
            pid = all_pids[j]
            orig_cam = all_points_3d[pid]['observations'][0][0]
            if (cam_idx + 1) != orig_cam:  # don't count the originating camera
                all_points_3d[pid]['observations'].append(
                    (cam_idx + 1, float(u_proj[j]), float(v_proj[j]))
                )

    # min_views 이상에서 관측된 points만 유지
    for pid in all_pids:
        if len(all_points_3d[pid]['observations']) >= min_views:
            final_points[point_id_new] = all_points_3d[pid]
            point_id_new += 1

    print(f"Sparse points: {len(all_points_3d)} → {len(final_points)} (>={min_views} views)")
    return final_points


# ─── COLMAP binary writers ───

def write_cameras_binary(cameras, path):
    """Write cameras.bin. cameras: dict of {cam_id: (model_id, width, height, params)}"""
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for cam_id, (model_id, width, height, params) in cameras.items():
            f.write(struct.pack('<iiQQ', cam_id, model_id, width, height))
            for p in params:
                f.write(struct.pack('<d', p))


def write_images_binary(images_data, path):
    """Write images.bin.
    images_data: dict of {image_id: (qvec, tvec, camera_id, name, xys, point3D_ids)}
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images_data)))
        for image_id, (qvec, tvec, camera_id, name, xys, point3D_ids) in images_data.items():
            f.write(struct.pack('<i', image_id))
            for q in qvec:
                f.write(struct.pack('<d', q))
            for t in tvec:
                f.write(struct.pack('<d', t))
            f.write(struct.pack('<i', camera_id))
            # name as null-terminated string
            f.write(name.encode('utf-8'))
            f.write(b'\x00')
            # 2D points
            f.write(struct.pack('<Q', len(xys)))
            for (x, y), pid in zip(xys, point3D_ids):
                f.write(struct.pack('<ddq', x, y, pid))


def write_points3D_binary(points, path):
    """Write points3D.bin.
    points: dict of {point_id: {xyz, rgb, error, observations: [(img_id, x, y)]}}
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for pid, pt in points.items():
            xyz = pt['xyz']
            rgb = pt['rgb']
            error = pt['error']
            obs = pt['observations']
            # point3D_id, x, y, z, r, g, b, error
            f.write(struct.pack('<Q', pid))
            f.write(struct.pack('<ddd', *xyz))
            f.write(struct.pack('<BBB', int(rgb[0]), int(rgb[1]), int(rgb[2])))
            f.write(struct.pack('<d', error))
            # track: num_elements, then (image_id, point2D_idx) pairs
            f.write(struct.pack('<Q', len(obs)))
            for (img_id, x, y) in obs:
                f.write(struct.pack('<ii', img_id, 0))  # point2D_idx=0 placeholder


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Directory with rgb_*.png, depth_raw_*.png, calib_*.ini')
    parser.add_argument('--output_dir', required=True, help='Output directory (COLMAP format)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--depth_subsample', type=int, default=50,
                        help='Pixel grid spacing for sparse point sampling from GT depth')
    parser.add_argument('--min_views', type=int, default=2,
                        help='Minimum views a point must be seen in')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    # Find all views
    calib_files = sorted(glob(os.path.join(input_dir, "calib_*.ini")))
    n_views = len(calib_files)
    print(f"Found {n_views} views")

    # Parse all calibrations
    calib_data = {}
    for idx in range(n_views):
        calib_data[idx] = parse_calib_ini(os.path.join(input_dir, f"calib_{idx:04d}.ini"))

    K0, _, _, width, height = calib_data[0]
    fx, fy, cx, cy = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    print(f"Image: {width}x{height}, fx={fx:.1f}, fy={fy:.1f}")

    # ─── Create output dirs ───
    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # ─── Copy images (rename to 00000.png format) ───
    print("Copying images...")
    for idx in tqdm(range(n_views)):
        src = os.path.join(input_dir, f"rgb_{idx:04d}.png")
        dst = os.path.join(images_dir, f"{idx:05d}.png")
        if not os.path.exists(dst):
            copy2(src, dst)

    # ─── Write cameras.bin (single PINHOLE camera) ───
    # PINHOLE model_id = 1, params = [fx, fy, cx, cy]
    cameras = {1: (1, width, height, [fx, fy, cx, cy])}
    write_cameras_binary(cameras, os.path.join(sparse_dir, "cameras.bin"))
    print("cameras.bin written")

    # ─── Generate sparse 3D points from GT depth ───
    sparse_points = create_sparse_points_from_depth(
        calib_data, input_dir, n_views,
        subsample_pixels=args.depth_subsample,
        min_views=args.min_views
    )

    # ─── Build per-image 2D-3D correspondences ───
    # For images.bin, each image needs xys and point3D_ids
    img_observations = {idx: {'xys': [], 'pids': []} for idx in range(n_views)}
    for pid, pt in sparse_points.items():
        for (img_id, x, y) in pt['observations']:
            img_idx = img_id - 1  # img_id is 1-based
            img_observations[img_idx]['xys'].append((x, y))
            img_observations[img_idx]['pids'].append(pid)

    # ─── Write images.bin ───
    images_bin_data = {}
    for idx in range(n_views):
        K, R, T, w, h = calib_data[idx]
        qvec = rotmat2qvec(R)  # R is w2c rotation
        tvec = T
        name = f"{idx:05d}.png"
        xys = np.array(img_observations[idx]['xys']) if img_observations[idx]['xys'] else np.zeros((0, 2))
        pids = np.array(img_observations[idx]['pids'], dtype=np.int64) if img_observations[idx]['pids'] else np.zeros(0, dtype=np.int64)
        images_bin_data[idx + 1] = (qvec, tvec, 1, name, xys, pids)

    write_images_binary(images_bin_data, os.path.join(sparse_dir, "images.bin"))
    print("images.bin written")

    # ─── Write points3D.bin ───
    write_points3D_binary(sparse_points, os.path.join(sparse_dir, "points3D.bin"))
    print(f"points3D.bin written ({len(sparse_points)} points)")

    # ─── Train/Test split ───
    split_path = os.path.join(output_dir, "split_index.json")
    if not os.path.exists(split_path):
        np.random.seed(args.split_seed)
        all_idx = list(range(n_views))
        np.random.shuffle(all_idx)
        n_train = int(args.train_ratio * n_views)
        train_idx = sorted(all_idx[:n_train])
        test_idx = sorted(all_idx[n_train:])
        with open(split_path, 'w') as f:
            json.dump({"train": train_idx, "test": test_idx}, f, indent=2)
        print(f"split_index.json: train={len(train_idx)}, test={len(test_idx)}")
    else:
        print("split_index.json already exists, skipping")

    print(f"\n변환 완료: {output_dir}")
    print("이제 저자의 원래 파이프라인으로 학습 가능:")
    print(f"  python train.py -s {output_dir} --eval --depth --usedepthReg ...")


if __name__ == "__main__":
    main()
