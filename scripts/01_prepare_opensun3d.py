import os, argparse, csv, glob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., /path/to/ChallengeDevelopmentSet")
    ap.add_argument("--out", required=True, help="output frames.csv")
    args = ap.parse_args()

    scene_dirs = sorted([p for p in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(p)])
    rows = []
    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)
        rgb_dir = os.path.join(scene_dir, "lowres_wide")
        depth_dir = os.path.join(scene_dir, "lowres_depth")
        intr_dir = os.path.join(scene_dir, "lowres_wide_intrinsics")
        traj = os.path.join(scene_dir, "lowres_wide.traj")
        if not os.path.isdir(rgb_dir): 
            continue
        for rp in sorted(glob.glob(os.path.join(rgb_dir, "*.png"))):
            ts = os.path.splitext(os.path.basename(rp))[0]
            dp = os.path.join(depth_dir, f"{ts}.png")
            ip = os.path.join(intr_dir, f"{ts}.pincam")
            if os.path.exists(dp) and os.path.exists(ip) and os.path.exists(traj):
                rows.append({"scene_id": scene_id, "timestamp": ts, "rgb_path": rp, "depth_path": dp, "intr_path": ip, "traj_path": traj})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["scene_id","timestamp","rgb_path","depth_path","intr_path","traj_path"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] wrote {args.out} rows={len(rows)} scenes={len(scene_dirs)}")

if __name__ == "__main__":
    main()
