from pathlib import Path

import copy
import joblib
import numpy as np
import pandas as pd
import open3d as o3d

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

DATA_DIR = BASE_DIR / "data"

OUTPUT_DIR = BASE_DIR / "output"

OUTPUT_DIR.mkdir(exist_ok=True)

model = joblib.load(MODEL_PATH)

print("AI Model Loaded Successfully")

def clean_mesh_topology(mesh):

    mesh.remove_duplicated_vertices()

    mesh.remove_duplicated_triangles()

    mesh.remove_degenerate_triangles()

    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    mesh.compute_triangle_normals()

    return mesh

def point_cloud_from_mesh(mesh, voxel_size):

    # Sample points from mesh
    triangle_count = len(mesh.triangles)

    sample_count = min(
        max(triangle_count * 3, 50000),
        200000
    )

    pcd = mesh.sample_points_uniformly(
        number_of_points=sample_count
    )

    # Downsample
    pcd = pcd.voxel_down_sample(
        voxel_size=voxel_size
    )

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5,
            max_nn=50
        )
    )

    return pcd

def make_watertight_poisson(mesh, params):

    if not mesh.has_triangles():
        return mesh

    mesh.compute_vertex_normals()

    # Convert mesh to point cloud
    pcd = mesh.sample_points_uniformly(
        number_of_points=100000
    )

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=params["voxel_size"] * 5,
            max_nn=50
        )
    )

    # Orient normals
    pcd.orient_normals_consistent_tangent_plane(30)

    # Poisson reconstruction
    poisson_mesh, density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=params["poisson_depth"]
    )

    densities = np.asarray(density)

    vertices_to_remove = densities < np.quantile(densities, 0.03)

    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    poisson_mesh.remove_unreferenced_vertices()

    poisson_mesh.compute_vertex_normals()

    # Reduce mesh size while preserving shape
    target_triangles = max(
        10000,
        int(len(poisson_mesh.triangles) * 0.60)
    )

    poisson_mesh = poisson_mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_triangles
    )

    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_unreferenced_vertices()

    poisson_mesh.compute_vertex_normals()
    poisson_mesh.compute_triangle_normals()

    return poisson_mesh

def laplacian_smooth(mesh, params):

    smoothed = mesh.filter_smooth_laplacian(

        number_of_iterations=params["laplacian_iters"]

    )

    smoothed.compute_vertex_normals()

    return smoothed

def taubin_smooth(mesh, params):

    smoothed = mesh.filter_smooth_taubin(

          number_of_iterations=params["taubin_iters"]

    )

    smoothed.compute_vertex_normals()

    return smoothed

def optimize_mesh(mesh, params):

    # Step 1 - Clean mesh
    optimized = clean_mesh_topology(copy.deepcopy(mesh))

    # Step 2 - Poisson reconstruction
    optimized = make_watertight_poisson(
        optimized,
        params
    )

    # Step 3 - Laplacian smoothing
    optimized = laplacian_smooth(
        optimized,
        params
    )

    # Step 4 - Taubin smoothing
    optimized = taubin_smooth(
        optimized,
        params
    )

    # Step 5 - Final mesh cleaning
    optimized = clean_mesh_topology(
        optimized
    )

    return optimized

file_name = input("Enter STL file name: ")

mesh_path = DATA_DIR / file_name

if not mesh_path.exists():
    print("File not found.")
    exit()

mesh = o3d.io.read_triangle_mesh(str(mesh_path))

if not mesh.has_triangles():
    print("Invalid STL file.")
    exit()

mesh.compute_vertex_normals()

vertices = len(mesh.vertices)

triangles = len(mesh.triangles)

surface_area = mesh.get_surface_area()

sample = pd.DataFrame([{

    "Vertices": vertices,

    "Triangles": triangles,

    "Surface_Area": surface_area

}])

prediction = model.predict(sample)

params = {

    "voxel_size": float(prediction[0][0]),

    "poisson_depth": int(round(prediction[0][1])),

    "laplacian_iters": int(round(prediction[0][2])),

    "taubin_iters": int(round(prediction[0][3]))

}

print("\nPredicted Parameters")
print(params)

optimized_mesh = optimize_mesh(
    mesh,
    params
)

output_name = file_name.replace(
    ".stl",
    "_optimized.stl"
)

output_path = OUTPUT_DIR / output_name

o3d.io.write_triangle_mesh(
    str(output_path),
    optimized_mesh
)

print("\nOptimization Completed Successfully")
print("Saved:", output_path)
   