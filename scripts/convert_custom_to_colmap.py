"""
calib_XXXX.ini + depth_raw_XXXX.png → COLMAP binary format 변환기.

Usage:
    python scripts/convert_custom_to_colmap.py \
        --input_dir data/mitsubishi_raw \
        --output_dir data/mitsubishi \
        --train_ratio 0.8 \
        --split_seed 42 \
        --depth_subsample 50 \
        --num_source_views 20
"""

import os
import struct
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
from shutil import copy2
import cv2


# ─── calib INI parser ───

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


def rotmat2qvec(R):
    """Rotation matrix → COLMAP quaternion (w, x, y, z)."""
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


# ─── Sparse 3D points generation ───

def create_sparse_points(calib_data, input_dir, n_views,
                         subsample_pixels=50, num_source_views=20):
    """
    소수의 source 뷰에서 GT depth를 back-project하여 3D points 생성 후,
    모든 뷰에 reproject하여 visibility를 결정합니다.

    Returns:
        points_3d: dict {point_id: {xyz, rgb, error, observations: [(img_id, x, y)]}}
                   모든 point는 ≥2 뷰에서 관측됨을 보장.
    """
    # source 뷰 균등 선택
    source_indices = np.linspace(0, n_views - 1, num_source_views, dtype=int)
    print(f"Source views for point generation: {len(source_indices)} views")

    K0 = calib_data[0][0]
    fx, fy = K0[0, 0], K0[1, 1]
    cx, cy = K0[0, 2], K0[1, 2]
    width, height = calib_data[0][3], calib_data[0][4]

    # ── Step 1: source 뷰에서 sparse depth → 3D points ──
    all_xyz = []
    all_rgb = []
    all_origin_cam = []  # 어느 뷰에서 생성되었는지

    for idx in tqdm(source_indices, desc="Back-projecting source views"):
        K, R, T, w, h = calib_data[idx]

        depth_path = os.path.join(input_dir, f"depth_raw_{idx:04d}.png")
        if not os.path.exists(depth_path):
            continue
        depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth_raw.astype(np.float64) * 0.01

        rgb_path = os.path.join(input_dir, f"rgb_{idx:04d}.png")
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) if rgb_img is not None else np.full((h, w, 3), 128, dtype=np.uint8)

        # subsample grid
        ys = np.arange(0, h, subsample_pixels)
        xs = np.arange(0, w, subsample_pixels)
        uu, vv = np.meshgrid(xs, ys)
        uu, vv = uu.ravel(), vv.ravel()
        dd = depth[vv, uu]
        mask = dd > 0
        uu, vv, dd = uu[mask], vv[mask], dd[mask]

        x_cam = (uu - cx) * dd / fx
        y_cam = (vv - cy) * dd / fy
        p_cam = np.stack([x_cam, y_cam, dd], axis=1)
        p_world = (R.T @ (p_cam.T - T.reshape(3, 1))).T

        all_xyz.append(p_world)
        all_rgb.append(rgb_img[vv, uu])
        all_origin_cam.extend([idx] * len(p_world))

    all_xyz = np.concatenate(all_xyz).astype(np.float64)
    all_rgb = np.concatenate(all_rgb)
    all_origin_cam = np.array(all_origin_cam)
    n_pts = len(all_xyz)
    print(f"Generated {n_pts} candidate 3D points")

    # ── Step 2: 모든 뷰에 reproject → visibility matrix (vectorized) ──
    # observations[point_idx] = list of (img_id_1based, u, v)
    observations = [[] for _ in range(n_pts)]

    for cam_idx in tqdm(range(n_views), desc="Reprojecting to all views"):
        K, R, T, w, h = calib_data[cam_idx]

        p_cam = (R @ all_xyz.T + T.reshape(3, 1))  # (3, N)
        z = p_cam[2]

        with np.errstate(divide='ignore', invalid='ignore'):
            u_proj = fx * p_cam[0] / z + cx
            v_proj = fy * p_cam[1] / z + cy

        valid = (z > 0) & (u_proj >= 0) & (u_proj < w) & (v_proj >= 0) & (v_proj < h)
        valid_indices = np.where(valid)[0]

        img_id = cam_idx + 1  # 1-based
        for j in valid_indices:
            observations[j].append((img_id, float(u_proj[j]), float(v_proj[j])))

    # ── Step 3: ≥2 뷰에서 관측된 points만 유지 ──
    points_3d = {}
    pid = 1  # 1-based point IDs
    for i in range(n_pts):
        if len(observations[i]) >= 2:
            points_3d[pid] = {
                'xyz': all_xyz[i],
                'rgb': all_rgb[i],
                'error': 1.0,
                'observations': observations[i],
            }
            pid += 1

    print(f"Final sparse points: {len(points_3d)} (≥2 views)")
    return points_3d


# ─── COLMAP binary writers ───

def write_cameras_binary(cameras, path):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(cameras)))
        for cam_id, (model_id, width, height, params) in cameras.items():
            f.write(struct.pack('<iiQQ', cam_id, model_id, width, height))
            for p in params:
                f.write(struct.pack('<d', p))


def write_images_binary(images_data, path):
    """
    images_data: dict {image_id: (qvec, tvec, camera_id, name, xys, point3D_ids)}
    xys: np.array (N, 2), point3D_ids: np.array (N,) — -1 for unmatched
    """
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(images_data)))
        for image_id in sorted(images_data.keys()):
            qvec, tvec, camera_id, name, xys, point3D_ids = images_data[image_id]
            f.write(struct.pack('<i', image_id))
            for q in qvec:
                f.write(struct.pack('<d', q))
            for t in tvec:
                f.write(struct.pack('<d', t))
            f.write(struct.pack('<i', camera_id))
            f.write(name.encode('utf-8'))
            f.write(b'\x00')
            n_pts = len(xys)
            f.write(struct.pack('<Q', n_pts))
            for k in range(n_pts):
                f.write(struct.pack('<ddq', xys[k, 0], xys[k, 1], int(point3D_ids[k])))


def write_points3D_binary(points, path):
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(points)))
        for pid, pt in sorted(points.items()):
            xyz = pt['xyz']
            rgb = pt['rgb']
            error = pt['error']
            obs = pt['observations']
            f.write(struct.pack('<Q', pid))
            f.write(struct.pack('<ddd', xyz[0], xyz[1], xyz[2]))
            f.write(struct.pack('<BBB', int(rgb[0]), int(rgb[1]), int(rgb[2])))
            f.write(struct.pack('<d', error))
            f.write(struct.pack('<Q', len(obs)))
            for (img_id, x, y) in obs:
                f.write(struct.pack('<ii', img_id, 0))


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--depth_subsample', type=int, default=50)
    parser.add_argument('--num_source_views', type=int, default=20)
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    calib_files = sorted(glob(os.path.join(input_dir, "calib_*.ini")))
    n_views = len(calib_files)
    print(f"Found {n_views} views")

    calib_data = {}
    for idx in range(n_views):
        calib_data[idx] = parse_calib_ini(os.path.join(input_dir, f"calib_{idx:04d}.ini"))

    K0, _, _, width, height = calib_data[0]
    fx, fy, cx, cy = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
    print(f"Image: {width}x{height}, fx={fx:.1f}, fy={fy:.1f}")

    # ── Output dirs ──
    images_dir = os.path.join(output_dir, "images")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    # ── Copy images ──
    print("Copying images...")
    for idx in tqdm(range(n_views)):
        src = os.path.join(input_dir, f"rgb_{idx:04d}.png")
        dst = os.path.join(images_dir, f"{idx:05d}.png")
        if not os.path.exists(dst):
            copy2(src, dst)

    # ── cameras.bin (single PINHOLE) ──
    cameras = {1: (1, width, height, [fx, fy, cx, cy])}
    write_cameras_binary(cameras, os.path.join(sparse_dir, "cameras.bin"))
    print("cameras.bin written")

    # ── Sparse 3D points ──
    sparse_points = create_sparse_points(
        calib_data, input_dir, n_views,
        subsample_pixels=args.depth_subsample,
        num_source_views=args.num_source_views
    )

    # ── Build per-image 2D-3D correspondences ──
    # 각 이미지가 관측하는 point의 2D 위치 + point3D_id 매핑
    img_obs = {idx: {'xys': [], 'pids': []} for idx in range(n_views)}
    for pid, pt in sparse_points.items():
        for (img_id, x, y) in pt['observations']:
            img_idx = img_id - 1
            img_obs[img_idx]['xys'].append((x, y))
            img_obs[img_idx]['pids'].append(pid)

    # ── images.bin ──
    # 핵심: 각 이미지에 최소 1개의 point3D_id=-1 더미 엔트리 추가.
    # refineColmapWithIndex가 unique(point3D_ids)[1:]로 -1을 제거하기 때문.
    images_bin_data = {}
    for idx in range(n_views):
        K, R, T, w, h = calib_data[idx]
        qvec = rotmat2qvec(R)
        tvec = T
        name = f"{idx:05d}.png"

        matched_xys = img_obs[idx]['xys']
        matched_pids = img_obs[idx]['pids']

        # 더미 unmatched 엔트리 추가 (point3D_id = -1)
        all_xys = [(0.0, 0.0)] + matched_xys
        all_pids = [-1] + matched_pids

        xys = np.array(all_xys, dtype=np.float64)
        pids = np.array(all_pids, dtype=np.int64)

        images_bin_data[idx + 1] = (qvec, tvec, 1, name, xys, pids)

    write_images_binary(images_bin_data, os.path.join(sparse_dir, "images.bin"))
    print(f"images.bin written ({n_views} images)")

    # ── points3D.bin ──
    write_points3D_binary(sparse_points, os.path.join(sparse_dir, "points3D.bin"))
    print(f"points3D.bin written ({len(sparse_points)} points)")

    # ── split_index.json ──
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
        print("split_index.json already exists")

    print(f"\n변환 완료: {output_dir}")
    print(f"  python train.py -s {output_dir} --eval --depth --usedepthReg ...")


if __name__ == "__main__":
    main()
