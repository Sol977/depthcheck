import os, argparse, glob
from pathlib import Path
import numpy as np
import open3d as o3d

from src.utils.geom import read_pincam
from src.utils.io import load_traj_map
from src.utils.traj_lookup import lookup_Twc


def raycast_depth(mesh_path: str, intr_path: str, Twc: np.ndarray) -> np.ndarray:
    """
    Raycast depth (meters) from a mesh given pinhole intrinsics and camera-to-world pose Twc.
    Returns (H,W) float32 depth, 0 for miss.
    """
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    if mesh_legacy.is_empty():
        raise RuntimeError(f"Empty mesh: {mesh_path}")

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    intr = read_pincam(intr_path)
    W, H = int(intr["width"]), int(intr["height"])
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])

    # pixel grid
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x = (uu - cx) / fx
    y = (vv - cy) / fy
    z = np.ones_like(x, dtype=np.float32)

    dirs_c = np.stack([x, y, z], axis=-1)  # (H,W,3)
    dirs_c /= np.clip(np.linalg.norm(dirs_c, axis=-1, keepdims=True), 1e-6, None)

    # camera-to-world: Xw = R Xc + t  => direction: dw = R dc
    R = Twc[:3, :3]
    t = Twc[:3, 3]

    dirs_w = dirs_c.reshape(-1, 3) @ R.T
    origins_w = np.repeat(t[None, :], dirs_w.shape[0], axis=0)

    rays = np.concatenate([origins_w, dirs_w], axis=1).astype(np.float32)
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)

    ans = scene.cast_rays(rays_t)
    t_hit = ans["t_hit"].numpy().reshape(H, W).astype(np.float32)
    t_hit[np.isinf(t_hit)] = 0.0
    return t_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=".../ChallengeDevelopmentSet")
    ap.add_argument("--scene_id", required=True)
    ap.add_argument("--out_dir", required=True, help="proxy_root")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3, help="timestamp match tolerance (seconds)")
    args = ap.parse_args()

    scene = Path(args.root) / args.scene_id
    traj_path = scene / "lowres_wide.traj"
    intr_dir = scene / "lowres_wide_intrinsics"
    rgb_dir = scene / "lowres_wide"

    # pick the 3dod mesh
    mesh_candidates = list(scene.glob(f"{args.scene_id}*_3dod_mesh.ply")) + list(scene.glob("*_3dod_mesh.ply"))
    if not mesh_candidates:
        raise FileNotFoundError("3dod mesh not found in scene directory")
    mesh_path = str(mesh_candidates[0])

    traj_map = load_traj_map(str(traj_path))
    # 기존 코드의 이 부분을 찾아서 아래 내용으로 바꿔서 실행해 보세요.

    rgb_paths = []
    for ext in ("png","jpg","jpeg","PNG","JPG","JPEG"):
        rgb_paths += glob.glob(str(rgb_dir / f"*.{ext}"))
    rgb_paths = sorted(rgb_paths)[::args.stride]

    print(f"[DEBUG] Found {len(rgb_paths)} RGB images in {rgb_dir}") # [체크 1] 이미지가 잡히는지 확인

    out_scene = Path(args.out_dir) / args.scene_id
    out_scene.mkdir(parents=True, exist_ok=True)

    used = 0
    for i, rp in enumerate(rgb_paths):
        if used >= args.max_frames:
            break
        ts_full = Path(rp).stem
        ts = ts_full.split('_')[-1]

        Twc = lookup_Twc(traj_map, ts, tol=args.tol)
        if Twc is None:
            # [체크 2] 첫 5개 실패 사례만 출력 (너무 많이 출력되는 것 방지)
            if i < 5: 
                print(f"[DEBUG] Traj lookup failed for ts={ts} (origin: {ts_full})")
            continue

        intr_path = intr_dir / f"{ts}.pincam"
        if not intr_path.exists():
            intr_path = intr_dir / f"{ts_full}.pincam" # 혹시 원본 이름 그대로일 경우 대비
            # [체크 3] 내부 파라미터 파일 부재
        if not intr_path.exists():
            if i < 5:
                print(f"[DEBUG] Intrinsics missing: {intr_path}")
            continue

        depth = raycast_depth(mesh_path, str(intr_path), Twc)
        np.save(out_scene / f"{ts}.npy", depth.astype(np.float32))
        used += 1

        if used % 20 == 0:
            print(f"[proxy] {used}/{args.max_frames}")

    print(f"[OK] proxy depth saved: {out_scene} frames={used}")

if __name__ == "__main__":
    main()
