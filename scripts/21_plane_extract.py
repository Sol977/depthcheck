import os, argparse, json
import numpy as np
import open3d as o3d

def fit_planes(mesh: o3d.geometry.TriangleMesh, max_planes: int=5, distance_threshold: float=0.02, ransac_n: int=3, num_iter: int=2000):
    """
    Plane model: ax + by + cz + d = 0, with ||(a,b,c)||=1 approximately.
    Returns list of dicts.
    """
    pcd = mesh.sample_points_uniformly(number_of_points=200000)
    pts = np.asarray(pcd.points)
    remaining = pcd
    planes = []
    for k in range(max_planes):
        if len(remaining.points) < 5000:
            break
        plane_model, inliers = remaining.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iter)
        inliers = np.array(inliers, dtype=np.int64)
        if inliers.size < 5000:
            break
        a,b,c,d = plane_model
        # normalize
        n = np.array([a,b,c], dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm > 1e-12:
            a,b,c,d = a/norm, b/norm, c/norm, d/norm
        inlier_pts = np.asarray(remaining.points)[inliers]
        centroid = inlier_pts.mean(axis=0).tolist()
        planes.append({
            "id": k,
            "abcd": [float(a),float(b),float(c),float(d)],
            "num_inliers": int(inliers.size),
            "centroid": centroid
        })
        remaining = remaining.select_by_index(inliers.tolist(), invert=True)
    return planes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_planes", type=int, default=8)
    ap.add_argument("--dist", type=float, default=0.03)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty():
        raise RuntimeError("Empty mesh")
    planes = fit_planes(mesh, max_planes=args.max_planes, distance_threshold=args.dist)
    out_json = os.path.join(args.out_dir, "planes.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"mesh": args.mesh, "planes": planes}, f, indent=2)
    print(f"[OK] planes={len(planes)} -> {out_json}")

if __name__ == "__main__":
    main()
