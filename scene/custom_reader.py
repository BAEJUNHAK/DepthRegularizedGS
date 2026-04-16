import os
import json
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import cv2

from utils.graphics_utils import BasicPointCloud, focal2fov
from scene.dataset_readers import (
    CameraInfo, SceneInfo, getNerfppNorm, storePly, fetchPly
)


def parse_calib_ini(filepath):
    """calib_XXXX.ini 파싱 → K(3x3), R(3x3 w2c), T(3,), width, height"""
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


def _load_depth(path, idx, w, h, resolution):
    """depth_raw_XXXX.png (uint16) → float32 meters. 없으면 None."""
    depth_path = os.path.join(path, f"depth_raw_{idx:04d}.png")
    if not os.path.exists(depth_path):
        return None
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth_raw.astype(np.float32) * 0.01  # uint16 → meters
    if resolution > 1:
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    return depth


def _mask_background(image_pil, depth):
    """depth==0 인 배경 픽셀을 검정(0,0,0)으로 마스킹.
    렌더링 bg_color=[0,0,0]과 일치시켜 photometric loss 낭비 방지.
    (이 데이터는 배경이 88.8%이므로 효과가 큼)
    """
    img_arr = np.array(image_pil)            # (H, W, 3) uint8
    fg_mask = (depth > 0) if depth is not None else np.ones(img_arr.shape[:2], dtype=bool)
    img_arr[~fg_mask] = 0                    # 배경 → 검정
    return Image.fromarray(img_arr, "RGB")


def readCustomBlenderSceneInfo(path, images, eval, kshot=1000, seed=0,
                                resolution=1, white_background=False):
    """calib INI + GT depth 기반 커스텀 Blender 데이터 리더.

    - 모든 뷰에서 GT depth mask로 배경을 검정 처리 (bg_color=[0,0,0] 일치)
    - train 뷰에만 depth supervision용 depth map 전달
    - 초기 point cloud는 GT depth back-projection으로 생성
    """

    # --- 뷰 수 확인 ---
    calib_files = sorted(glob(os.path.join(path, "calib_*.ini")))
    n_views = len(calib_files)
    print(f"Found {n_views} views in {path}")

    # --- Split 로드 ---
    split_path = os.path.join(path, "split_index.json")
    with open(split_path) as f:
        split = json.load(f)
    train_idx = split["train"]
    test_idx = split["test"]

    # --- K-shot 샘플링 ---
    np.random.seed(seed)
    if eval and kshot < len(train_idx):
        train_idx = sorted(
            np.random.choice(train_idx, size=kshot, replace=False).tolist()
        )
    print(f"Train: {len(train_idx)} views, Test: {len(test_idx)} views")

    # --- 모든 카메라 로드 ---
    cam_infos = []
    for idx in tqdm(range(n_views), desc="Loading cameras"):
        calib_path = os.path.join(path, f"calib_{idx:04d}.ini")
        K, R_w2c, T_w2c, width, height = parse_calib_ini(calib_path)

        w = width // resolution
        h = height // resolution
        fx = K[0, 0] / resolution
        fy = K[1, 1] / resolution

        FovX = focal2fov(fx, w)
        FovY = focal2fov(fy, h)

        # RGB
        rgb_path = os.path.join(path, f"rgb_{idx:04d}.png")
        image = Image.open(rgb_path).convert("RGB")
        if resolution > 1:
            image = image.resize((w, h))

        # Depth: 모든 뷰에서 로드 (배경 마스킹용)
        depth_full = _load_depth(path, idx, w, h, resolution)

        # 배경 제거: depth==0 픽셀을 검정으로 (bg_color=[0,0,0] 일치)
        image = _mask_background(image, depth_full)

        # depth supervision용 depth는 train 뷰에만 전달
        depth_for_supervision = depth_full if idx in train_idx else None

        # DRGS convention: R stored as transpose of w2c rotation
        R_stored = np.transpose(R_w2c)

        cam_infos.append(CameraInfo(
            uid=idx, R=R_stored, T=T_w2c,
            FovY=FovY, FovX=FovX,
            image=image, depth=depth_for_supervision, depth_weight=None,
            image_path=rgb_path, image_name=f"{idx:04d}",
            width=w, height=h, depthloss=0.0
        ))

    # --- Train/Test split ---
    if eval:
        train_cam_infos = [cam_infos[i] for i in train_idx]
        test_cam_infos = [cam_infos[i] for i in test_idx]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # --- Point cloud from GT depth back-projection ---
    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        print("Creating point cloud from GT depth...")
        all_pts, all_cols = [], []

        for ci in tqdm(train_cam_infos, desc="Back-projecting"):
            if ci.depth is None:
                continue
            h_img, w_img = ci.depth.shape
            K_c, _, _, _, _ = parse_calib_ini(
                os.path.join(path, f"calib_{ci.uid:04d}.ini"))
            fx_c = K_c[0, 0] / resolution
            fy_c = K_c[1, 1] / resolution
            cx_c = K_c[0, 2] / resolution
            cy_c = K_c[1, 2] / resolution

            R_stored = ci.R      # = R_w2c.T
            T_w2c = ci.T

            u, v = np.meshgrid(np.arange(w_img), np.arange(h_img))
            d = ci.depth
            mask = d > 0
            uf, vf, df = u[mask], v[mask], d[mask]

            # pixel → camera coords
            x_c = (uf - cx_c) * df / fx_c
            y_c = (vf - cy_c) * df / fy_c
            p_cam = np.stack([x_c, y_c, df], axis=1)  # (N, 3)

            # camera → world: p_world = R_stored @ (p_cam - T)
            p_world = (R_stored @ (p_cam.T - T_w2c.reshape(3, 1))).T

            cols = np.array(ci.image)[vf, uf, :3]
            all_pts.append(p_world.astype(np.float32))
            all_cols.append(cols)

        all_pts = np.concatenate(all_pts)
        all_cols = np.concatenate(all_cols)

        # Subsample to 100K
        if len(all_pts) > 100_000:
            sel = np.random.choice(len(all_pts), 100_000, replace=False)
            all_pts, all_cols = all_pts[sel], all_cols[sel]

        storePly(ply_path, all_pts, all_cols)
        print(f"Point cloud: {len(all_pts)} pts → {ply_path}")

    pcd = fetchPly(ply_path)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path
    )
