# RGB2BIM Starter Kit v2 (Teacher–Student + proxy GT + IFCWallStandardCase)

This v2 extends the baseline to a **paper-ready experiment structure**:

- **Proxy depth GT from ARKitScenes/OpenSUN3D 3DOD mesh** (raycast)  
  → lets you supervise a teacher without needing the full laser-scan package.
- **Teacher (RGB + sensor depth) → refined depth**  
- **Student (RGB only) distillation** from teacher (+ optional proxy GT supervision)
- **TSDF fusion** using student depth + known poses
- **Plane patches → boundary hull → parametric IFC export** (IfcWallStandardCase/IfcSlab)

> Why proxy GT? The OpenSUN3D challenge subset includes a reconstructed mesh (`*_3dod_mesh.ply`).
We raycast this mesh to get a deterministic depth target for each frame.

## A) Install
```bash
conda create -n rgb2bim python=3.10 -y
conda activate rgb2bim
pip install -r requirements.txt
```

## B) Prepare OpenSUN3D frame list
```bash
python scripts/01_prepare_opensun3d.py --root /path/to/ChallengeDevelopmentSet --out data/frames.csv
```

## C) Build proxy GT depth (raycast mesh -> depth)
```bash
python scripts/05_raycast_mesh_to_depth.py --root /path/to/ChallengeDevelopmentSet --scene_id 42445173 \
  --out_dir data/proxy_depth --stride 3 --max_frames 300
```

This writes: `data/proxy_depth/<scene_id>/<timestamp>.npy` (float32 meters).

## D) Train teacher (RGB + sensor depth -> refined depth)
```bash
python scripts/12_train_teacher.py --frames_csv data/frames.csv --proxy_root data/proxy_depth --out_dir runs/teacher
```

## E) Train student (RGB only) via distillation (+ optional proxy supervision)
```bash
python scripts/13_train_student_distill.py --frames_csv data/frames.csv --proxy_root data/proxy_depth \
  --teacher_ckpt runs/teacher/best.pt --out_dir runs/student_distill
```

## F) Fuse a scene with student depth (TSDF)
```bash
python scripts/20_tsdf_fuse_scene.py --root /path/to/ChallengeDevelopmentSet --scene_id 42445173 \
  --checkpoint runs/student_distill/best.pt --out_dir out/tsdf
```

## G) Extract planes + boundary hulls
```bash
python scripts/21_plane_extract_hulls.py --mesh out/tsdf/42445173_mesh.ply --out_dir out/planes
```

## H) Export IFC (IfcWallStandardCase + IfcSlab)
```bash
python scripts/22_export_ifc_parametric.py --planes_json out/planes/planes.json --out_ifc out/planes/model.ifc
```

## I) Unity/Blender synthetic -> OpenSUN3D-like format
If you can export per-frame `{timestamp, K, Twc}` and RGB/depth images, convert with:
```bash
python scripts/02_unity_to_opensun3d.py --in_dir /path/to/unity_export --out_scene /path/to/output/00000000
```

Expected `in_dir`:
- `frames.json` with list of frames (timestamp, width,height, fx,fy,cx,cy, Twc 4x4)
- `rgb/<timestamp>.png`
- `depth_m/<timestamp>.npy` (float32 meters) OR `depth_mm/<timestamp>.png` (uint16 mm)

---

## Notes for paper
- Report depth metrics against **proxy GT depth** and show BIM/IFC geometric error against proxy mesh (point-to-mesh).
- The IFC exporter here is still a baseline; you will extend it with:
  - wall topology stitching (adjacency graph)
  - openings (IfcOpeningElement + feature.add_feature)
  - room polygons

