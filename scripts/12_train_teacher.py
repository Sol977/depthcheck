import os, argparse, csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from src.models.unet_depth4 import UNetDepth4
from src.losses import masked_l1, masked_si_log
from src.utils.io import load_rgb, load_depth_meters

class TeacherDataset(Dataset):
    def __init__(self, frames_csv: str, proxy_root: str, max_frames=None):
        self.rows = []
        # 디버깅을 위한 카운터
        checked = 0
        found = 0
        
        with open(frames_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # [수정된 부분] 타임스탬프에서 SceneID 제거 (예: 42445173_8098.138 -> 8098.138)
                ts_short = r["timestamp"].split('_')[-1]
                
                # 수정된 타임스탬프로 경로 생성
                p = os.path.join(proxy_root, r["scene_id"], f'{ts_short}.npy')
                
                checked += 1
                if os.path.exists(p):
                    r["proxy_path"] = p
                    self.rows.append(r)
                    found += 1
        
        # [디버그 출력] 실제로 몇 개를 찾았는지 확인
        print(f"[Dataset] Checked {checked} rows from CSV. Found {found} matching .npy files.")
        
        if found == 0:
            print(f"[Warning] No matching files found. Check path: {proxy_root}/<scene_id>/<timestamp>.npy")

        if max_frames is not None:
            self.rows = self.rows[:max_frames]

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        rgb = load_rgb(r["rgb_path"])
        sensor = load_depth_meters(r["depth_path"])  # meters
        proxy = np.load(r["proxy_path"]).astype(np.float32)

        rgb_t = torch.from_numpy(rgb).float().permute(2,0,1) / 255.0
        sensor_t = torch.from_numpy(sensor).float().unsqueeze(0)
        x = torch.cat([rgb_t, sensor_t], dim=0)  # 4ch

        gt = torch.from_numpy(proxy).float().unsqueeze(0)
        valid = (gt > 1e-6).float()
        return {"x": x, "gt": gt, "valid": valid}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_csv", required=True)
    ap.add_argument("--proxy_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--max_frames", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    ds = TeacherDataset(args.frames_csv, args.proxy_root, max_frames=args.max_frames)
    n_val = max(1, int(len(ds)*args.val_ratio))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(0))

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetDepth4(in_ch=4, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"teacher train {epoch}/{args.epochs}")
        for batch in pbar:
            x = batch["x"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            pred = model(x)
            l1 = masked_l1(pred, gt, valid)
            sil = masked_si_log(pred, gt, valid)
            loss = l1 + 0.1*sil
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            pbar.set_postfix({"loss": float(loss.item())})

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["x"].to(device, non_blocking=True)
                gt = batch["gt"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                pred = model(x)
                loss = masked_l1(pred, gt, valid) + 0.1*masked_si_log(pred, gt, valid)
                va_loss += float(loss.item())
        va_loss /= max(1, len(dl_va))

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_loss": va_loss}
        torch.save(ckpt, out/"last.pt")
        if va_loss < best:
            best = va_loss
            torch.save(ckpt, out/"best.pt")
        print(f"[epoch {epoch}] val_loss={va_loss:.6f} best={best:.6f}")

if __name__ == "__main__":
    main()
