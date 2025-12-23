import os, argparse, csv, glob, bisect
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.models.unet_depth import UNetDepth
from src.models.unet_depth4 import UNetDepth4
from src.losses import masked_l1
from src.utils.io import load_rgb, load_depth_meters


class DistillDataset(Dataset):
    """
    Uses ONLY frames that have matching proxy depth:
      frames.csv timestamp = "{scene_id}_{t}"
      proxy file name      = "{t}.npy"
    """
    def __init__(self, frames_csv: str, proxy_root: str, max_frames=None, proxy_tol: float=0.05):
        self.rows = []

        # build proxy index: scene_id -> sorted (t_float, path)
        proxy_index = {}
        for sd in [d for d in glob.glob(os.path.join(proxy_root, "*")) if os.path.isdir(d)]:
            sid = os.path.basename(sd)
            items = []
            for fp in glob.glob(os.path.join(sd, "*.npy")):
                t_str = os.path.splitext(os.path.basename(fp))[0]
                try:
                    items.append((float(t_str), fp))
                except Exception:
                    continue
            items.sort(key=lambda x: x[0])
            proxy_index[sid] = items

        def strip_prefix(scene_id: str, ts: str) -> str:
            pref = scene_id + "_"
            return ts[len(pref):] if ts.startswith(pref) else ts

        def nearest_proxy(scene_id: str, ts_raw: str):
            items = proxy_index.get(scene_id, [])
            if not items:
                return None
            t_str = strip_prefix(scene_id, ts_raw)
            try:
                t = float(t_str)
            except Exception:
                return None
            t_list = [x[0] for x in items]
            j = bisect.bisect_left(t_list, t)
            cand = []
            for jj in (j-1, j, j+1):
                if 0 <= jj < len(items):
                    cand.append((abs(items[jj][0] - t), items[jj][1]))
            if not cand:
                return None
            d, fp = min(cand, key=lambda x: x[0])
            return fp if d <= proxy_tol else None

        with open(frames_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                fp = nearest_proxy(r["scene_id"], r["timestamp"])
                if fp is None:
                    continue
                r["proxy_path"] = fp
                self.rows.append(r)

        if max_frames is not None:
            self.rows = self.rows[:max_frames]

        print(f"[Dataset] Found {len(self.rows)} distill frames with proxy depth.")

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        rgb = load_rgb(r["rgb_path"])                    # uint8 HWC
        sensor = load_depth_meters(r["depth_path"])      # float32 meters
        proxy = np.load(r["proxy_path"]).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
        sensor_t = torch.from_numpy(sensor).float().unsqueeze(0)
        x_teacher = torch.cat([rgb_t, sensor_t], dim=0)  # 4ch

        proxy_t = torch.from_numpy(proxy).float().unsqueeze(0)
        valid_t = (proxy_t > 1e-6).float()
        return {"rgb": rgb_t, "x_teacher": x_teacher, "proxy": proxy_t, "valid": valid_t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_csv", required=True)
    ap.add_argument("--proxy_root", required=True)
    ap.add_argument("--teacher_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--lambda_proxy", type=float, default=0.3)
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ds = DistillDataset(args.frames_csv, args.proxy_root, max_frames=args.max_frames)
    if len(ds) == 0:
        raise RuntimeError("DistillDataset is empty. Check proxy_root and frames.csv timestamp format.")

    n_val = max(1, int(len(ds)*args.val_ratio))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(0))
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("[Device]", device)

    teacher = UNetDepth4(in_ch=4, base=32).to(device)
    tckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(tckpt["model"], strict=True)
    teacher.eval()

    student = UNetDepth(in_ch=3, base=32).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr)

    best = 1e9
    for epoch in range(1, args.epochs+1):
        student.train()
        pbar = tqdm(dl_tr, desc=f"student distill {epoch}/{args.epochs}")
        for batch in pbar:
            rgb = batch["rgb"].to(device, non_blocking=True)
            x_teacher = batch["x_teacher"].to(device, non_blocking=True)
            proxy = batch["proxy"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            with torch.no_grad():
                tpred = teacher(x_teacher)

            spred = student(rgb)
            loss_distill = (spred - tpred).abs().mean()
            loss_proxy = masked_l1(spred, proxy, valid)
            loss = loss_distill + args.lambda_proxy * loss_proxy

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": float(loss.item()), "distill": float(loss_distill.item()), "proxy": float(loss_proxy.item())})

        student.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in dl_va:
                rgb = batch["rgb"].to(device, non_blocking=True)
                x_teacher = batch["x_teacher"].to(device, non_blocking=True)
                tpred = teacher(x_teacher)
                spred = student(rgb)
                va_loss += float((spred - tpred).abs().mean().item())
        va_loss /= max(1, len(dl_va))

        ckpt = {"model": student.state_dict(), "epoch": epoch, "val_loss": va_loss}
        torch.save(ckpt, out/"last.pt")
        if va_loss < best:
            best = va_loss
            torch.save(ckpt, out/"best.pt")
        print(f"[epoch {epoch}] val_loss(distill)={va_loss:.6f} best={best:.6f}")

if __name__ == "__main__":
    main()
