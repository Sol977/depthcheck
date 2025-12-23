import os, argparse, json
import numpy as np
import open3d as o3d

def pca_axes(pts: np.ndarray):
    c = pts.mean(axis=0)
    X = pts - c
    C = (X.T @ X) / max(1, X.shape[0])
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    return c, V[:, order], w[order]

def project_to_plane(pts, n, origin):
    # build orthonormal basis (u,v) on plane
    n = n / (np.linalg.norm(n) + 1e-12)
    tmp = np.array([1.0,0.0,0.0])
    if abs(np.dot(tmp, n)) > 0.9:
        tmp = np.array([0.0,1.0,0.0])
    u = np.cross(n, tmp); u = u/(np.linalg.norm(u)+1e-12)
    v = np.cross(n, u); v = v/(np.linalg.norm(v)+1e-12)
    X = pts - origin[None,:]
    uv = np.stack([X @ u, X @ v], axis=1)  # (N,2)
    return uv, u, v

def convex_hull_2d(points):
    # Monotonic chain; points: (N,2)
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    def cross(o,a,b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower=[]
    for p in pts:
        while len(lower)>=2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper=[]
    for p in pts[::-1]:
        while len(upper)>=2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=np.float32)

def fit_planes_with_inliers(mesh, max_planes=10, dist=0.03, ransac_n=3, iters=2000):
    pcd = mesh.sample_points_uniformly(number_of_points=250000)
    remaining = pcd
    planes=[]
    for k in range(max_planes):
        if len(remaining.points) < 8000:
            break
        model, inliers = remaining.segment_plane(distance_threshold=dist, ransac_n=ransac_n, num_iterations=iters)
        inliers = np.array(inliers, dtype=np.int64)
        if inliers.size < 8000:
            break
        a,b,c,d = model
        n = np.array([a,b,c], dtype=np.float64)
        nn = np.linalg.norm(n)
        if nn>1e-12:
            a,b,c,d = a/nn, b/nn, c/nn, d/nn
        pts = np.asarray(remaining.points)[inliers]
        centroid = pts.mean(axis=0)
        uv, u, v = project_to_plane(pts, np.array([a,b,c],dtype=np.float64), centroid)
        hull_uv = convex_hull_2d(uv.astype(np.float32))
        # map hull back to 3D
        hull_xyz = centroid[None,:] + hull_uv[:,0:1]*u[None,:] + hull_uv[:,1:2]*v[None,:]
        # bbox extents in plane
        min_uv = uv.min(axis=0); max_uv = uv.max(axis=0)
        planes.append({
            "id": int(k),
            "abcd": [float(a),float(b),float(c),float(d)],
            "num_inliers": int(inliers.size),
            "centroid": centroid.tolist(),
            "basis_u": u.tolist(),
            "basis_v": v.tolist(),
            "hull_uv": hull_uv.tolist(),
            "hull_xyz": hull_xyz.tolist(),
            "extent_uv": [float(min_uv[0]), float(min_uv[1]), float(max_uv[0]), float(max_uv[1])]
        })
        remaining = remaining.select_by_index(inliers.tolist(), invert=True)
    return planes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_planes", type=int, default=12)
    ap.add_argument("--dist", type=float, default=0.03)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(args.mesh)
    if mesh.is_empty():
        raise RuntimeError("Empty mesh")
    planes = fit_planes_with_inliers(mesh, max_planes=args.max_planes, dist=args.dist)
    out_json = os.path.join(args.out_dir, "planes.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"mesh": args.mesh, "planes": planes}, f, indent=2)
    print(f"[OK] planes={len(planes)} -> {out_json}")

if __name__ == "__main__":
    main()
