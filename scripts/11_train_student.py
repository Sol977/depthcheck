import os, argparse, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.datasets.frames_csv import FramesCSVDepthDataset
from src.models.unet_depth import UNetDepth
from src.losses import masked_l1, masked_si_log

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    ds = FramesCSVDepthDataset(args.frames_csv, max_frames=args.max_frames)
    n_val = max(1, int(len(ds) * args.val_ratio))
    n_tr = len(ds) - n_val
    tr, va = random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(0))

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetDepth(in_ch=3, base=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 1e9
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"train {epoch}/{args.epochs}")
        tr_loss = 0.0
        for batch in pbar:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            mask = batch["valid"].to(device, non_blocking=True)

            pred = model(rgb)
            l1 = masked_l1(pred, gt, mask)
            sil = masked_si_log(pred, gt, mask)
            loss = l1 + 0.1 * sil

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in dl_va:
                rgb = batch["rgb"].to(device, non_blocking=True)
                gt = batch["depth"].to(device, non_blocking=True)
                mask = batch["valid"].to(device, non_blocking=True)
                pred = model(rgb)
                l1 = masked_l1(pred, gt, mask)
                sil = masked_si_log(pred, gt, mask)
                loss = l1 + 0.1 * sil
                va_loss += loss.item()
        va_loss /= max(1, len(dl_va))

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_loss": va_loss}
        torch.save(ckpt, out/"last.pt")
        if va_loss < best:
            best = va_loss
            torch.save(ckpt, out/"best.pt")

        print(f"[epoch {epoch}] train_loss={tr_loss/max(1,len(dl_tr)):.6f} val_loss={va_loss:.6f} best={best:.6f}")

if __name__ == "__main__":
    main()
