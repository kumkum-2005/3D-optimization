import os
from pathlib import Path
import pandas as pd
import numpy as np
import open3d as o3d

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATASET_DIR = BASE_DIR / "dataset"

DATASET_DIR.mkdir(exist_ok=True)
rows = []

stl_files = list(DATA_DIR.glob("*.stl"))

if len(stl_files) == 0:
    print("No STL files found in data folder.")
    exit()

print(f"Found {len(stl_files)} STL files\n")

for file in stl_files:

    print(f"Processing: {file.name}")

    try:
        mesh = o3d.io.read_triangle_mesh(str(file))

        if not mesh.has_triangles():
            print("Skipped (Invalid STL)")
            continue

        mesh.compute_vertex_normals()

        vertices = len(mesh.vertices)
        triangles = len(mesh.triangles)

        surface_area = mesh.get_surface_area()

        try:
            volume = mesh.get_volume()
        except:
            volume = 0

        bbox = mesh.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()

        width = extent[0]
        height = extent[1]
        depth = extent[2]

        file_size = os.path.getsize(file)

        rows.append({
            "File_Name": file.name,
            "Vertices": vertices,
            "Triangles": triangles,
            "Surface_Area": round(surface_area,4),
            "Volume": round(volume,4),
            "Width": round(width,4),
            "Height": round(height,4),
            "Depth": round(depth,4),
            "File_Size": file_size
        })

    except Exception as e:
        print("Error:", e)

df = pd.DataFrame(rows)

csv_path = DATASET_DIR / "mesh_dataset.csv"

df.to_csv(csv_path, index=False)

print("\n===================================")
print("Dataset Created Successfully")
print(csv_path)
print("Total Models:", len(df))
print("===================================")