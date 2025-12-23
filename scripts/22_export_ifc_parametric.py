import argparse, json
from pathlib import Path
import numpy as np
import ifcopenshell
import ifcopenshell.api


def api_run_compat(model, usecase: str, **kw):
    try:
        return ifcopenshell.api.run(usecase, model, **kw)
    except TypeError as e:
        # products vs product
        if usecase in ("aggregate.assign_object", "spatial.assign_container", "geometry.edit_object_placement"):
            if "product" in kw and "products" not in kw:
                prod = kw.pop("product")
                kw["products"] = [prod]
                return ifcopenshell.api.run(usecase, model, **kw)
        raise e


def assign_units_meters(model):
    try:
        api_run_compat(model, "unit.assign_unit", length={"unit": "METRE"})
    except Exception:
        pass


def add_basic_spatial_structure(model):
    project = api_run_compat(model, "root.create_entity", ifc_class="IfcProject", name="RGB2BIM")
    assign_units_meters(model)

    context = api_run_compat(model, "context.add_context", context_type="Model")
    body = api_run_compat(
        model, "context.add_context",
        context_type="Model", context_identifier="Body",
        target_view="MODEL_VIEW", parent=context
    )

    site = api_run_compat(model, "root.create_entity", ifc_class="IfcSite", name="Site")
    building = api_run_compat(model, "root.create_entity", ifc_class="IfcBuilding", name="Building")
    storey = api_run_compat(model, "root.create_entity", ifc_class="IfcBuildingStorey", name="Level 0")

    # default placements
    api_run_compat(model, "geometry.edit_object_placement", product=site)
    api_run_compat(model, "geometry.edit_object_placement", product=building)
    api_run_compat(model, "geometry.edit_object_placement", product=storey)

    api_run_compat(model, "aggregate.assign_object", relating_object=project, products=[site])
    api_run_compat(model, "aggregate.assign_object", relating_object=site, products=[building])
    api_run_compat(model, "aggregate.assign_object", relating_object=building, products=[storey])

    return project, site, building, storey, body


def polygon_area_xy(poly):
    pts = poly[:]
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    a = 0.0
    for i in range(len(pts)-1):
        x1,y1 = pts[i]
        x2,y2 = pts[i+1]
        a += x1*y2 - x2*y1
    return 0.5*a


def ensure_ccw(poly):
    return list(reversed(poly)) if polygon_area_xy(poly) < 0 else poly


def make_profile_uv(model, hull_uv):
    # hull_uv: list of [u,v]
    poly = [[float(u), float(v)] for u,v in hull_uv]
    poly = ensure_ccw(poly)
    if poly[0] != poly[-1]:
        poly.append(poly[0])

    pts = [model.create_entity("IfcCartesianPoint", (p[0], p[1])) for p in poly]
    polyline = model.create_entity("IfcPolyline", pts)
    profile = model.create_entity("IfcArbitraryClosedProfileDef", "AREA", None, polyline)
    return profile


def make_extruded_representation(model, body_context, hull_uv, height):
    profile = make_profile_uv(model, hull_uv)

    # local origin of swept solid at (0,0,0)
    origin = model.create_entity("IfcCartesianPoint", (0.0, 0.0, 0.0))
    pos = model.create_entity("IfcAxis2Placement3D", origin, None, None)
    dirz = model.create_entity("IfcDirection", (0.0, 0.0, 1.0))

    solid = model.create_entity("IfcExtrudedAreaSolid", profile, pos, dirz, float(height))
    shape = model.create_entity("IfcShapeRepresentation", body_context, "Body", "SweptSolid", [solid])
    rep = model.create_entity("IfcProductDefinitionShape", None, None, [shape])
    return rep


def make_local_placement(model, relto, origin_xyz, x_dir, y_dir, n_hint=None):
    import numpy as np
    x = np.asarray(x_dir, dtype=np.float64)
    y = np.asarray(y_dir, dtype=np.float64)
    x = x / (np.linalg.norm(x) + 1e-12)
    # Gram-Schmidt: make y orthogonal to x
    y = y - np.dot(y, x) * x
    y = y / (np.linalg.norm(y) + 1e-12)
    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + 1e-12)

    if n_hint is not None:
        n = np.asarray(n_hint, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)
        # align z to plane normal hemisphere
        if np.dot(z, n) < 0:
            z = -z
            y = -y  # keep right-handed: cross(x,y)=z

    ox, oy, oz = map(float, origin_xyz)
    p = model.create_entity("IfcCartesianPoint", (ox, oy, oz))
    axis = model.create_entity("IfcDirection", tuple(map(float, z.tolist())))
    ref  = model.create_entity("IfcDirection", tuple(map(float, x.tolist())))
    a2p  = model.create_entity("IfcAxis2Placement3D", p, axis, ref)
    return model.create_entity("IfcLocalPlacement", relto, a2p)



def iter_planes(j):
    if isinstance(j, dict) and isinstance(j.get("planes"), list):
        return j["planes"]
    if isinstance(j, list):
        return j
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes_json", required=True)
    ap.add_argument("--out_ifc", required=True)
    ap.add_argument("--wall_thickness", type=float, default=0.15)
    ap.add_argument("--slab_thickness", type=float, default=0.20)
    ap.add_argument("--slab_nz_thresh", type=float, default=0.85)
    args = ap.parse_args()

    j = json.loads(Path(args.planes_json).read_text(encoding="utf-8"))
    model = ifcopenshell.file(schema="IFC4")
    _, _, _, storey, body = add_basic_spatial_structure(model)

    n_added = 0
    for pl in iter_planes(j):
        hull_uv = pl.get("hull_uv", None)
        if hull_uv is None or len(hull_uv) < 3:
            continue

        abcd = pl.get("abcd", None)
        centroid = pl.get("centroid", None)
        bu = pl.get("basis_u", None)
        bv = pl.get("basis_v", None)
        if abcd is None or centroid is None or bu is None or bv is None:
            continue

        a,b,c,d = map(float, abcd)
        n = np.array([a,b,c], dtype=np.float64)
        n_norm = np.linalg.norm(n) + 1e-12
        n = n / n_norm

        centroid = np.array(list(map(float, centroid)), dtype=np.float64)
        bu = np.array(list(map(float, bu)), dtype=np.float64)
        bv = np.array(list(map(float, bv)), dtype=np.float64)
        bu = bu / (np.linalg.norm(bu) + 1e-12)
        # 로컬 Z는 평면 법선(=extrude 방향), 로컬 X는 basis_u
        z_dir = n
        x_dir = bu

        # slab vs wall 분류: |nz| 큰 평면 = 수평면(슬래브)
        thickness = args.slab_thickness if abs(z_dir[2]) > args.slab_nz_thresh else args.wall_thickness

        rep = make_extruded_representation(model, body, hull_uv, thickness)
        cls = "IfcSlab" if abs(z_dir[2]) > args.slab_nz_thresh else "IfcWall"
        prod = api_run_compat(model, "root.create_entity", ifc_class=cls, name=f"{cls}_{pl.get('id','')}")
        prod.Representation = rep

        # 로컬 배치(세계좌표에서 centroid로 이동 + 축 정렬)
        prod.ObjectPlacement = make_local_placement(
    model,
    storey.ObjectPlacement,
    centroid,
    bu, bv,
    n_hint=n
)


        api_run_compat(model, "spatial.assign_container", products=[prod], relating_structure=storey)
        n_added += 1

    out_path = Path(args.out_ifc)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.write(str(out_path))
    print(f"[OK] wrote {out_path} elements={n_added}")


if __name__ == "__main__":
    main()
