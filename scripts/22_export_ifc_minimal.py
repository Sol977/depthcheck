import argparse, json
import ifcopenshell
import ifcopenshell.api
import numpy as np

def unit_normal(abcd):
    n = np.array(abcd[:3], dtype=float)
    nn = np.linalg.norm(n)
    return n/nn if nn>1e-12 else n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planes_json", required=True)
    ap.add_argument("--out_ifc", required=True)
    args = ap.parse_args()

    with open(args.planes_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    planes = data["planes"]

    f = ifcopenshell.file(schema="IFC4")
    project = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcProject", name="RGB2BIM")
    ifcopenshell.api.run("unit.assign_unit", f, length_units="METERS")
    context = ifcopenshell.api.run("context.add_context", f, context_type="Model")
    site = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    building = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuilding", name="Building")
    storey = ifcopenshell.api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name="Storey 0")

    ifcopenshell.api.run("aggregate.assign_object", f, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", f, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", f, relating_object=building, products=[storey])

    # Minimal geometry: create thin slabs/walls as generic extrusions around centroid (placeholder).
    # You will replace this with proper boundary reconstruction later.
    def make_rect_profile(width=2.0, height=2.0):
        return ifcopenshell.api.run("profile.create_profile", f, profile_type="AREA", profile_name="Rect",
                                    x_dim=width, y_dim=height)

    def place_at(x,y,z):
        return ifcopenshell.api.run("geometry.add_local_coordinate_system", f, origin=(x,y,z))

    for pl in planes:
        cx,cy,cz = pl["centroid"]
        n = unit_normal(pl["abcd"])
        # classify very roughly by normal direction (no assumptions beyond dot-products)
        up = abs(float(n[2]))
        if up > 0.9:
            ifc_class = "IfcSlab"
            name = f"Slab_{pl['id']}"
            thickness = 0.15
        else:
            ifc_class = "IfcWall"
            name = f"Wall_{pl['id']}"
            thickness = 0.2

        elem = ifcopenshell.api.run("root.create_entity", f, ifc_class=ifc_class, name=name)
        ifcopenshell.api.run("spatial.assign_container", f, relating_structure=storey, products=[elem])

        profile = make_rect_profile(width=3.0, height=0.2 if ifc_class=="IfcWall" else 3.0)
        rep = ifcopenshell.api.run(
            "geometry.add_profile_representation", f,
            context=context, profile=profile, depth=thickness
        )
        ifcopenshell.api.run("geometry.assign_representation", f, product=elem, representation=rep)
        ifcopenshell.api.run("geometry.edit_object_placement", f, product=elem, matrix=place_at(cx,cy,cz))

    f.write(args.out_ifc)
    print(f"[OK] wrote {args.out_ifc} elements={len(planes)}")

if __name__ == "__main__":
    main()
