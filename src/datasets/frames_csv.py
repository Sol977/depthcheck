import csv
from typing import Optional, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from ..utils.io import load_rgb, load_depth_meters
from ..utils.geom import read_pincam

class FramesCSVDepthDataset(Dataset):
    """
    Each row:
      scene_id,timestamp,rgb_path,depth_path,intr_path,traj_path
    Returns:
      rgb: float32 (3,H,W) in [0,1]
      depth: float32 (1,H,W) in meters
      valid_mask: float32 (1,H,W) (depth>0)
      intr: dict (fx,fy,cx,cy,w,h)
    """
    def __init__(self, frames_csv: str, max_frames: Optional[int]=None):
        self.rows = []
        with open(frames_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append(r)
        if max_frames is not None:
            self.rows = self.rows[:max_frames]

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        rgb = load_rgb(r["rgb_path"])
        depth = load_depth_meters(r["depth_path"])
        rgb_t = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
        depth_t = torch.from_numpy(depth).float().unsqueeze(0)
        valid = (depth_t > 1e-6).float()
        intr = read_pincam(r["intr_path"])
        return {"rgb": rgb_t, "depth": depth_t, "valid": valid, "intr": intr, "scene_id": r["scene_id"], "timestamp": r["timestamp"]}
