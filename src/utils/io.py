import os, glob
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import cv2

from .geom import traj_line_to_Twc, read_pincam

@dataclass
class FrameRecord:
    scene_id: str
    timestamp: str
    rgb_path: str
    depth_path: str
    intr_path: str
    traj_path: str

def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_depth_meters(path: str) -> np.ndarray:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    # OpenSUN3D depth is uint16 in millimeters (per their documentation)
    depth_m = depth.astype(np.float32) / 1000.0
    return depth_m

def load_traj_map(traj_path: str) -> Dict[str, np.ndarray]:
    """
    Returns map: timestamp_str -> T_wc (camera-to-world) 4x4

    IMPORTANT:
    OpenSUN3D/ARKitScenes timestamps in filenames keep many decimals.
    If we re-format floats (e.g., %.8f), matching will fail and frames will be skipped.
    So we store the RAW timestamp string from the .traj file as the primary key,
    plus a few normalized variants for robustness.
    """
    from .geom import rodrigues
    m: Dict[str, np.ndarray] = {}
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            ts_str = parts[0]
            r = np.array(list(map(float, parts[1:4])), dtype=np.float64)  # axis-angle (rad)
            t = np.array(list(map(float, parts[4:7])), dtype=np.float64)  # meters
            R = rodrigues(r)
            Twc = np.eye(4, dtype=np.float64)
            Twc[:3, :3] = R
            Twc[:3, 3] = t

            keys = set()
            keys.add(ts_str)
            try:
                ts = float(ts_str)
                keys.add(str(ts))  # python shortest repr
                keys.add(f"{ts:.8f}".rstrip("0").rstrip("."))
                keys.add(f"{ts:.11f}".rstrip("0").rstrip("."))
            except Exception:
                pass

            for k in keys:
                m[k] = Twc
    return m


def list_scene_frames(scene_dir: str) -> List[FrameRecord]:
    scene_id = os.path.basename(scene_dir.rstrip("/"))
    rgb_dir = os.path.join(scene_dir, "lowres_wide")
    depth_dir = os.path.join(scene_dir, "lowres_depth")
    intr_dir = os.path.join(scene_dir, "lowres_wide_intrinsics")
    traj_path = os.path.join(scene_dir, "lowres_wide.traj")
    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(rgb_dir)

    rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    recs: List[FrameRecord] = []
    for rp in rgb_paths:
        ts = os.path.splitext(os.path.basename(rp))[0]
        dp = os.path.join(depth_dir, f"{ts}.png")
        ip = os.path.join(intr_dir, f"{ts}.pincam")
        if not (os.path.exists(dp) and os.path.exists(ip) and os.path.exists(traj_path)):
            continue
        recs.append(FrameRecord(scene_id=scene_id, timestamp=ts, rgb_path=rp, depth_path=dp, intr_path=ip, traj_path=traj_path))
    return recs
