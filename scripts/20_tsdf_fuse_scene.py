import os, argparse, glob
from pathlib import Path
import numpy as np
import torch
import open3d as o3d

from src.models.unet_depth import UNetDepth
from src.utils.io import load_rgb
from src.utils.geom import read_pincam
from src.utils.io import load_traj_map
from src.utils.traj_lookup import lookup_Twc

def collect_images(rgb_dir: Path):
    rgb_paths = []
    for ext in ("png","jpg","jpeg","PNG","JPG","JPEG"):
        rgb_paths += glob.glob(str(rgb_dir / f"*.{ext}"))
    return sorted(rgb_paths)

def build_intr_index(intr_dir: Path):
    items = []
    for p in glob.glob(str(intr_dir / "*.pincam")):
        stem = Path(p).stem
        try:
            items.append((float(stem.split("_")[-1]), p))  # allow "scene_ts" or "ts"
        except Exception:
            continue
    items.sort(key=lambda x: x[0])
    ts_arr = np.array([t for t,_ in items], dtype=np.float64)
    paths = [p for _,p in items]
    return ts_arr, paths

def nearest_path(ts_arr: np.ndarray, paths: list, ts: float, tol: float):
    if ts_arr.size == 0:
        return None
    j = int(np.searchsorted(ts_arr, ts))
    cand = []
    for jj in (j-1, j, j+1):
        if 0 <= jj < ts_arr.size:
            cand.append((abs(float(ts_arr[jj]-ts)), paths[jj]))
    if not cand:
        return None
    d, p = min(cand, key=lambda x: x[0])
    return p if d <= tol else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scene_id", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--traj_tol", type=float, default=0.05)
    ap.add_argument("--intr_tol", type=float, default=0.05)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--trunc", type=float, default=0.09)
    ap.add_argument("--depth_trunc", type=float, default=6.0)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    scene = Path(args.root) / args.scene_id
    rgb_dir = scene / "lowres_wide"
    intr_dir = scene / "lowres_wide_intrinsics"
    traj_path = scene / "lowres_wide.traj"

    rgb_paths = collect_images(rgb_dir)[::args.stride]
    print(f"[DEBUG] rgb_found={len(rgb_paths)} dir={rgb_dir}")

    traj_map = load_traj_map(str(traj_path))
    intr_ts, intr_paths = build_intr_index(intr_dir)

    # model
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("[Device]", device)

    model = UNetDepth(in_ch=3, base=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    used = 0
    skipped_pose = 0
    skipped_intr = 0

    for i, rp in enumerate(rgb_paths):
        if used >= args.max_frames:
            break

        ts_full = Path(rp).stem
        ts = ts_full.split("_")[-1]  # "scene_ts" -> "ts", "ts" -> "ts"
        try:
            ts_f = float(ts)
        except Exception:
            continue

        Twc = lookup_Twc(traj_map, ts, tol=args.traj_tol)
        if Twc is None:
            skipped_pose += 1
            if skipped_pose <= 3:
                print(f"[DEBUG] pose miss: ts_full={ts_full} ts={ts}")
            continue

        intr_path = (intr_dir / f"{ts_full}.pincam")
        if not intr_path.exists():
            intr_path = (intr_dir / f"{ts}.pincam")
        if not intr_path.exists():
            # nearest intrinsics by float timestamp
            p = nearest_path(intr_ts, intr_paths, ts_f, tol=args.intr_tol)
            if p is None:
                skipped_intr += 1
                if skipped_intr <= 3:
                    print(f"[DEBUG] intr miss: ts_full={ts_full} ts={ts}")
                continue
            intr_path = Path(p)

        intr = read_pincam(str(intr_path))
        W, H = int(intr["width"]), int(intr["height"])
        fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

        # RGB -> depth (meters)
        rgb = load_rgb(str(rp))  # HWC uint8
        rgb_t = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0) / 255.0
        rgb_t = rgb_t.to(device, non_blocking=True)
        with torch.no_grad():
            depth = model(rgb_t)[0,0].detach().float().cpu().numpy()  # HxW meters (our model outputs positive)
        depth = np.clip(depth, 0.0, args.depth_trunc).astype(np.float32)

        color_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
        depth_o3d = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False
        )

        # Open3D integrate expects an extrinsic; many pipelines use world->camera.
        # If your geometry looks mirrored later, swap to np.linalg.inv(Twc).
        extrinsic = np.linalg.inv(Twc)

        volume.integrate(rgbd, intrinsic, extrinsic)
        used += 1
        if used % 20 == 0:
            print(f"[tsdf] integrated {used}/{args.max_frames}")

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_mesh = out_dir / f"{args.scene_id}_mesh.ply"
    o3d.io.write_triangle_mesh(str(out_mesh), mesh)
    print(f"[OK] saved {out_mesh} (frames_used={used})")
    if used == 0:
        print(f"[WARN] frames_used=0 pose_miss={skipped_pose} intr_miss={skipped_intr} rgb_found={len(rgb_paths)}")

if __name__ == "__main__":
    main()
