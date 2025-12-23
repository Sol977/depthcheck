import numpy as np

def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle (rx,ry,rz) in radians -> 3x3 rotation matrix."""
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = axis_angle / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R

def traj_line_to_Twc(line: str) -> tuple[float, np.ndarray]:
    """
    OpenSUN3D `.traj` line format (space-delimited):
      timestamp  rx ry rz  tx ty tz
    where rotation is axis-angle (radians), translation is meters.
    Returns timestamp, T_wc (camera-to-world) 4x4.
    """
    parts = line.strip().split()
    ts = float(parts[0])
    r = np.array(list(map(float, parts[1:4])), dtype=np.float64)
    t = np.array(list(map(float, parts[4:7])), dtype=np.float64)
    R = rodrigues(r)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return ts, T

def read_pincam(path: str) -> dict:
    """
    `.pincam` single-line format:
      width height fx fy cx cy
    """
    with open(path, "r", encoding="utf-8") as f:
        parts = f.read().strip().split()
    w, h = int(float(parts[0])), int(float(parts[1]))
    fx, fy, cx, cy = map(float, parts[2:6])
    return {"width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}

def to_o3d_intrinsics(p: dict):
    import open3d as o3d
    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(int(p["width"]), int(p["height"]), float(p["fx"]), float(p["fy"]), float(p["cx"]), float(p["cy"]))
    return intr
