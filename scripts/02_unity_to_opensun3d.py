import os, argparse, json
from pathlib import Path
import numpy as np
import cv2

from src.utils.rot import mat_to_axis_angle

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_scene", required=True, help="output scene folder (e.g., .../00000000)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_scene = Path(args.out_scene)
    (out_scene/"lowres_wide").mkdir(parents=True, exist_ok=True)
    (out_scene/"lowres_depth").mkdir(parents=True, exist_ok=True)
    (out_scene/"lowres_wide_intrinsics").mkdir(parents=True, exist_ok=True)

    frames = json.loads((in_dir/"frames.json").read_text(encoding="utf-8"))
    traj_lines = []

    for fr in frames:
        ts = str(fr["timestamp"])
        w = int(fr["width"]); h = int(fr["height"])
        fx,fy,cx,cy = map(float, (fr["fx"], fr["fy"], fr["cx"], fr["cy"]))
        Twc = np.array(fr["Twc"], dtype=float).reshape(4,4)

        # rgb
        rgb_path = in_dir/"rgb"/f"{ts}.png"
        img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(rgb_path)
        cv2.imwrite(str(out_scene/"lowres_wide"/f"{ts}.png"), img)

        # depth
        depth_mm_path = in_dir/"depth_mm"/f"{ts}.png"
        depth_m_path = in_dir/"depth_m"/f"{ts}.npy"
        if depth_mm_path.exists():
            d = cv2.imread(str(depth_mm_path), cv2.IMREAD_UNCHANGED)
            if d is None:
                raise FileNotFoundError(depth_mm_path)
            cv2.imwrite(str(out_scene/"lowres_depth"/f"{ts}.png"), d)
        elif depth_m_path.exists():
            d = np.load(str(depth_m_path)).astype(np.float32)
            d_mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(str(out_scene/"lowres_depth"/f"{ts}.png"), d_mm)
        else:
            raise FileNotFoundError(f"Need depth_mm/{ts}.png or depth_m/{ts}.npy")

        # intr (.pincam)
        (out_scene/"lowres_wide_intrinsics"/f"{ts}.pincam").write_text(
            f"{w} {h} {fx} {fy} {cx} {cy}\n", encoding="utf-8"
        )

        # traj line: timestamp rx ry rz tx ty tz  (axis-angle radians, meters)
        R = Twc[:3,:3]
        t = Twc[:3,3]
        aa = mat_to_axis_angle(R)
        traj_lines.append(f"{ts} {aa[0]} {aa[1]} {aa[2]} {t[0]} {t[1]} {t[2]}\n")

    (out_scene/"lowres_wide.traj").write_text("".join(traj_lines), encoding="utf-8")
    print(f"[OK] wrote OpenSUN3D-like scene to {out_scene} frames={len(frames)}")

if __name__ == "__main__":
    main()
