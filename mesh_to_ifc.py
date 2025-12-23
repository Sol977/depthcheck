#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np

import open3d as o3d
import ifcopenshell


# --------------------------
# small utils
# --------------------------
def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def convex_hull_2d(points):
    """Monotonic chain convex hull. points: (N,2) -> hull (CCW, no repeated last)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts.tolist()
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        p = p.tolist()
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        p = p.tolist()
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

def polygon_area_2d(poly):
    if len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly], dtype=float)
    y = np.array([p[1] for p in poly], dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

def ensure_ccw(poly):
    if len(poly) < 3:
        return poly
    x = np.array([p[0] for p in poly], dtype=float)
    y = np.array([p[1] for p in poly], dtype=float)
    signed = 0.5 * (np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))
    return list(reversed(poly)) if signed < 0 else poly

def build_rotation_to_z_up(up_axis: str):
    """
    Return a right-handed orthonormal basis (ex,ey,ez) in original coords,
    such that ez aligns with chosen up axis; points are rotated as:
      p' = [p·ex, p·ey, p·ez]  (so p' uses z-up)
    """
    if up_axis == "x":
        ez = np.array([1,0,0], float)
    elif up_axis == "y":
        ez = np.array([0,1,0], float)
    else:
        ez = np.array([0,0,1], float)

    ref = np.array([1,0,0], float)
    if abs(np.dot(ref, ez)) > 0.9:
        ref = np.array([0,1,0], float)

    ex = normalize(np.cross(ref, ez))
    ey = normalize(np.cross(ez, ex))
    ez = normalize(ez)
    return ex, ey, ez

def rotate_points(P, ex, ey, ez):
    # P: (N,3) original -> (N,3) z-up
    return np.stack([P @ ex, P @ ey, P @ ez], axis=1)


# --------------------------
# IFC minimal builder
# --------------------------
def rectify_floor_to_pca_rect(xy):
    """
    xy: (N,2) floor inlier points
    return: 4 corners (CCW) of oriented rectangle in XY
    """
    xy = np.asarray(xy, float)
    c = xy.mean(axis=0)
    Q = xy - c
    C = (Q.T @ Q) / max(1, len(Q)-1)
    w, V = np.linalg.eigh(C)         # eigenvectors
    ex = V[:, np.argmax(w)]          # major axis
    ey = np.array([-ex[1], ex[0]])   # orthogonal

    u = Q @ ex
    v = Q @ ey
    u0, u1 = float(u.min()), float(u.max())
    v0, v1 = float(v.min()), float(v.max())

    corners = np.array([
        c + ex*u0 + ey*v0,
        c + ex*u1 + ey*v0,
        c + ex*u1 + ey*v1,
        c + ex*u0 + ey*v1,
    ], float)

    # ensure CCW
    return ensure_ccw(corners.tolist())

def make_owner_history(f):
    person = f.create_entity("IfcPerson", Identification=None, FamilyName="User", GivenName="SyntheticLiDAR")
    org = f.create_entity("IfcOrganization", Identification=None, Name="Lab", Description=None)
    p_o = f.create_entity("IfcPersonAndOrganization", ThePerson=person, TheOrganization=org)
    app = f.create_entity("IfcApplication", ApplicationDeveloper=org, Version="1.0",
                          ApplicationFullName="mesh_to_ifc", ApplicationIdentifier="mesh_to_ifc")
    owner_history = f.create_entity("IfcOwnerHistory", OwningUser=p_o, OwningApplication=app,
                                    State=None, ChangeAction="ADDED",
                                    LastModifiedDate=None, LastModifyingUser=None,
                                    LastModifyingApplication=None, CreationDate=0)
    return owner_history

def make_contexts(f, owner_history):
    project = f.create_entity("IfcProject", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
                              Name="Mesh2IFC", Description=None, ObjectType=None,
                              LongName=None, Phase=None, RepresentationContexts=None, UnitsInContext=None)

    # metres (if your mesh isn't in metres, use --scale)
    unit_length = f.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE", Prefix=None)
    unit_angle  = f.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN", Prefix=None)
    unit_area   = f.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE", Prefix=None)
    unit_vol    = f.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE", Prefix=None)
    project.UnitsInContext = f.create_entity("IfcUnitAssignment", Units=[unit_length, unit_angle, unit_area, unit_vol])

    wcs = f.create_entity(
        "IfcAxis2Placement3D",
        Location=f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
        Axis=f.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
        RefDirection=f.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0)),
    )
    ctx = f.create_entity("IfcGeometricRepresentationContext", ContextIdentifier="Body", ContextType="Model",
                          CoordinateSpaceDimension=3, Precision=1e-5, WorldCoordinateSystem=wcs, TrueNorth=None)
    project.RepresentationContexts = [ctx]

    site_placement = f.create_entity("IfcLocalPlacement", PlacementRelTo=None, RelativePlacement=wcs)
    site = f.create_entity("IfcSite", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, Name="Site",
                           ObjectPlacement=site_placement, Representation=None,
                           LongName=None, CompositionType="ELEMENT",
                           RefLatitude=None, RefLongitude=None, RefElevation=None,
                           LandTitleNumber=None, SiteAddress=None)

    building_placement = f.create_entity("IfcLocalPlacement", PlacementRelTo=site_placement, RelativePlacement=wcs)
    building = f.create_entity("IfcBuilding", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, Name="Building",
                               ObjectPlacement=building_placement, Representation=None,
                               LongName=None, CompositionType="ELEMENT",
                               ElevationOfRefHeight=None, ElevationOfTerrain=None, BuildingAddress=None)

    storey_placement = f.create_entity("IfcLocalPlacement", PlacementRelTo=building_placement, RelativePlacement=wcs)
    storey = f.create_entity("IfcBuildingStorey", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history, Name="Storey 0",
                             ObjectPlacement=storey_placement, Representation=None,
                             LongName=None, CompositionType="ELEMENT", Elevation=0.0)

    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
                    Name=None, Description=None, RelatingObject=project, RelatedObjects=[site])
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
                    Name=None, Description=None, RelatingObject=site, RelatedObjects=[building])
    f.create_entity("IfcRelAggregates", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner_history,
                    Name=None, Description=None, RelatingObject=building, RelatedObjects=[storey])

    return project, ctx, storey

def ifc_axis2placement3d(f, origin, xdir, zdir):
    origin = tuple(float(x) for x in origin)
    xdir = normalize(xdir)
    zdir = normalize(zdir)
    return f.create_entity("IfcAxis2Placement3D",
        Location=f.create_entity("IfcCartesianPoint", Coordinates=origin),
        Axis=f.create_entity("IfcDirection", DirectionRatios=tuple(float(x) for x in zdir)),
        RefDirection=f.create_entity("IfcDirection", DirectionRatios=tuple(float(x) for x in xdir)),
    )

def ifc_polyline_2d(f, pts2):
    pts = [f.create_entity("IfcCartesianPoint", Coordinates=(float(x), float(y))) for x,y in pts2]
    if pts2[0] != pts2[-1]:
        pts.append(f.create_entity("IfcCartesianPoint", Coordinates=(float(pts2[0][0]), float(pts2[0][1]))))
    return f.create_entity("IfcPolyline", Points=pts)

def add_extruded_solid(f, ctx, placement3d, profile_pts_2d, extrude_dir, depth):
    curve = ifc_polyline_2d(f, profile_pts_2d + [profile_pts_2d[0]])
    profile = f.create_entity("IfcArbitraryClosedProfileDef", ProfileType="AREA", ProfileName=None, OuterCurve=curve)
    solid = f.create_entity("IfcExtrudedAreaSolid",
        SweptArea=profile,
        Position=placement3d,
        ExtrudedDirection=f.create_entity("IfcDirection", DirectionRatios=tuple(float(x) for x in extrude_dir)),
        Depth=float(depth),
    )
    shape = f.create_entity("IfcProductDefinitionShape", Representations=[
        f.create_entity("IfcShapeRepresentation", ContextOfItems=ctx,
                        RepresentationIdentifier="Body", RepresentationType="SweptSolid", Items=[solid])
    ])
    return shape

def relate_to_storey(f, owner_history, storey, element):
    f.create_entity("IfcRelContainedInSpatialStructure", GlobalId=ifcopenshell.guid.new(),
                    OwnerHistory=owner_history, Name=None, Description=None,
                    RelatedElements=[element], RelatingStructure=storey)


# --------------------------
# Plane extraction + IFC
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_ply", required=True)
    ap.add_argument("--out_ifc", required=True)

    ap.add_argument("--scale", type=float, default=1.0, help="Multiply mesh coords by this factor")
    ap.add_argument("--up_axis", choices=["x","y","z"], default="z", help="Which axis is vertical in the input mesh")

    ap.add_argument("--sample_points", type=int, default=250000)
    ap.add_argument("--voxel", type=float, default=0.03)
    ap.add_argument("--stat_nb", type=int, default=30)
    ap.add_argument("--stat_std", type=float, default=2.0)

    ap.add_argument("--dbscan_eps", type=float, default=0.10)
    ap.add_argument("--dbscan_min", type=int, default=80)

    ap.add_argument("--dist_thresh", type=float, default=0.03)
    ap.add_argument("--min_inliers", type=int, default=4000)
    ap.add_argument("--max_planes", type=int, default=25)

    ap.add_argument("--min_floor_area", type=float, default=1.0)

    ap.add_argument("--wall_thickness", type=float, default=0.12)
    ap.add_argument("--slab_thickness", type=float, default=0.15)
    ap.add_argument("--max_walls", type=int, default=12)

    args = ap.parse_args()

    if not os.path.exists(args.mesh_ply):
        raise SystemExit(f"mesh not found: {args.mesh_ply}")

    # Read as mesh if possible, else point cloud
    mesh = o3d.io.read_triangle_mesh(args.mesh_ply)
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        pcd = o3d.io.read_point_cloud(args.mesh_ply)
        if len(pcd.points) == 0:
            raise SystemExit("Failed to read mesh or point cloud from ply")
    else:
        # Sample points from mesh surface
        pcd = mesh.sample_points_uniformly(number_of_points=args.sample_points)

    # Scale
    if args.scale != 1.0:
        pcd.scale(args.scale, center=(0.0,0.0,0.0))

    # Rotate to z-up basis (so IFC uses z-up)
    ex, ey, ez = build_rotation_to_z_up(args.up_axis)
    P = np.asarray(pcd.points)
    Pz = rotate_points(P, ex, ey, ez)
    pcd.points = o3d.utility.Vector3dVector(Pz)

    # Downsample + clean
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    if len(pcd.points) < args.min_inliers:
        raise SystemExit(f"Too few points after downsample: {len(pcd.points)}")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=args.stat_nb, std_ratio=args.stat_std)

    # Keep largest DBSCAN cluster (removes floating junk)
    labels = np.array(pcd.cluster_dbscan(eps=args.dbscan_eps, min_points=args.dbscan_min, print_progress=False))
    if labels.size > 0 and np.any(labels >= 0):
        vals, cnts = np.unique(labels[labels >= 0], return_counts=True)
        keep = vals[np.argmax(cnts)]
        idx = np.where(labels == keep)[0].tolist()
        pcd = pcd.select_by_index(idx)

    print(f"[INFO] points_for_ransac = {len(pcd.points)}")

    # Iterative plane segmentation
    planes = []
    rest = pcd
    for k in range(args.max_planes):
        if len(rest.points) < args.min_inliers:
            break
        model, inliers = rest.segment_plane(distance_threshold=args.dist_thresh, ransac_n=3, num_iterations=2000)
        if len(inliers) < args.min_inliers:
            break
        a,b,c,d = model
        n = normalize([a,b,c])
        inlier_cloud = rest.select_by_index(inliers)
        inlier_pts = np.asarray(inlier_cloud.points)

        # area proxy: convex hull on XY if horizontal-ish, else on XZ (length*height proxy)
        nz = abs(float(n[2]))
        if nz > 0.85:
            hull = convex_hull_2d(inlier_pts[:, :2])
            area = polygon_area_2d(hull)
            z_med = float(np.median(inlier_pts[:,2]))
        else:
            # wall-ish: use bbox extent product as proxy
            bbox = inlier_cloud.get_axis_aligned_bounding_box()
            ext = bbox.get_extent()
            area = float(max(ext[0], ext[1]) * ext[2])  # rough (length * height)
            z_med = float(np.median(inlier_pts[:,2]))

        planes.append({
            "model": (float(a),float(b),float(c),float(d)),
            "n": n,
            "nz": nz,
            "inliers": len(inliers),
            "area": area,
            "z_med": z_med,
            "pts": inlier_pts
        })
        rest = rest.select_by_index(inliers, invert=True)

    if not planes:
        raise SystemExit("No planes extracted. Try larger --dist_thresh or smaller --min_inliers")

    # Pick floor / ceiling from horizontal planes
    horiz = [p for p in planes if p["nz"] > 0.85]
    if not horiz:
        raise SystemExit("No horizontal planes found. Try different --up_axis (x/y/z).")

    horiz = sorted(horiz, key=lambda x: x["z_med"])
    floor = horiz[0]
    ceiling = horiz[-1] if len(horiz) >= 2 else None

    floor_z = float(floor["z_med"])
    # height fallback: use 95th percentile of all points
    all_z = np.asarray(pcd.points)[:,2]
    height_fallback = float(np.percentile(all_z, 95) - floor_z)
    if height_fallback <= 0:
        height_fallback = 2.6  # just a fallback number in "mesh units"

    if ceiling is not None:
        ceil_z = float(ceiling["z_med"])
        height = max(0.5, ceil_z - floor_z)
    else:
        ceil_z = floor_z + height_fallback
        height = max(0.5, height_fallback)

    # Floor outline (convex hull in XY)
    floor_xy = floor["pts"][:, :2]
    ffloor_poly = rectify_floor_to_pca_rect(floor_xy)
    if polygon_area_2d(floor_hull) < args.min_floor_area:
        raise SystemExit("Floor hull too small. Reduce --min_floor_area or check mesh scale/axis.")

    # Select wall planes (vertical-ish) by area, take top K
    walls = [p for p in planes if p["nz"] < 0.25]
    walls = sorted(walls, key=lambda x: (x["area"], x["inliers"]), reverse=True)[:args.max_walls]

    print(f"[INFO] planes={len(planes)} horiz={len(horiz)} walls_selected={len(walls)} floor_z={floor_z:.3f} ceil_z={ceil_z:.3f} height={height:.3f}")

    # Build IFC
    f = ifcopenshell.file(schema="IFC4")
    owner = make_owner_history(f)
    _, ctx, storey = make_contexts(f, owner)

    # Floor slab
    slab_place = ifc_axis2placement3d(f, origin=(0,0,floor_z), xdir=(1,0,0), zdir=(0,0,1))
    slab_shape = add_extruded_solid(f, ctx, slab_place, profile_pts_2d=floor_poly, extrude_dir=(0,0,1), depth=args.slab_thickness)
    slab = f.create_entity("IfcSlab", GlobalId=ifcopenshell.guid.new(), OwnerHistory=owner, Name="Floor",
                           Description=None, ObjectType=None,
                           ObjectPlacement=f.create_entity("IfcLocalPlacement", PlacementRelTo=storey.ObjectPlacement, RelativePlacement=slab_place),
                           Representation=slab_shape, Tag=None, PredefinedType="FLOOR")
    relate_to_storey(f, owner, storey, slab)

    # Ceiling slab (optional)
    if ceiling is not None:
        ceil_xy = ceiling["pts"][:, :2]
        ceil_hull = ensure_ccw(convex_hull_2d(ceil_xy))
        ceil_place = ifc_axis2placement3d(f, origin=(0,0,ceil_z - args.slab_thickness), xdir=(1,0,0), zdir=(0,0,1))
        ceil_shape = add_extruded_solid(f, ctx, ceil_place, ceil_hull, extrude_dir=(0,0,1), depth=args.slab_thickness)
        ceil_elem = f.create_entity(
            "IfcCovering",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name="Ceiling",
            Description=None,
            ObjectType=None,
            ObjectPlacement=f.create_entity(
                "IfcLocalPlacement",
                PlacementRelTo=storey.ObjectPlacement,
                RelativePlacement=ceil_place
            ),
            Representation=ceil_shape,
            Tag=None,
            PredefinedType="CEILING"
        )
        relate_to_storey(f, owner, storey, ceil_elem)


# ---- Walls from floor polygon edges (robust) ----
    t = float(args.wall_thickness)
    z_axis = np.array([0,0,1], float)

    poly = floor_poly  # 4 corners (CCW)
    wall_count = 0
    for i in range(len(poly)):
        p0 = np.array([poly[i][0], poly[i][1]], float)
        p1 = np.array([poly[(i+1)%len(poly)][0], poly[(i+1)%len(poly)][1]], float)
        e2 = p1 - p0
        L = float(np.linalg.norm(e2))
        if L < 0.2:
            continue

        xdir = np.array([e2[0]/L, e2[1]/L, 0.0], float)

        # place at edge start, on floor
        place = ifc_axis2placement3d(f, origin=(float(p0[0]), float(p0[1]), float(floor_z)), xdir=xdir, zdir=z_axis)

        # wall profile in local XY: length x thickness (centered)
        prof = [(0.0, -t/2), (L, -t/2), (L, t/2), (0.0, t/2)]
        shape = add_extruded_solid(f, ctx, place, prof, extrude_dir=(0,0,1), depth=height)

        wall = f.create_entity(
            "IfcWall",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=owner,
            Name=f"Wall_{wall_count}",
            Description=None,
            ObjectType=None,
            ObjectPlacement=f.create_entity("IfcLocalPlacement", PlacementRelTo=storey.ObjectPlacement, RelativePlacement=place),
            Representation=shape,
            Tag=None
        )
        relate_to_storey(f, owner, storey, wall)
        wall_count += 1


    os.makedirs(os.path.dirname(args.out_ifc) or ".", exist_ok=True)
    f.write(args.out_ifc)
    print(f"[OK] wrote IFC: {args.out_ifc} (walls={wall_count})")


if __name__ == "__main__":
    main()
