import numpy as np

def mat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert 3x3 rotation matrix to axis-angle (rx,ry,rz) where vector direction is axis and norm is angle (radians).
    """
    eps = 1e-12
    tr = np.trace(R)
    cos_theta = (tr - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < eps:
        return np.zeros(3, dtype=np.float64)
    # robust axis
    axis = np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ], dtype=np.float64)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < eps:
        # near pi; fallback from diagonal
        axis = np.sqrt(np.maximum(np.diag(R)+1.0, 0.0))
        axis = axis / (np.linalg.norm(axis) + eps)
    else:
        axis = axis / axis_norm
    return axis * theta
