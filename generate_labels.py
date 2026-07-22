import copy
from pathlib import Path

import numpy as np
import pandas as pd
import open3d as o3d

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"

DATASET_PATH = BASE_DIR / "dataset" / "mesh_dataset.csv"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATASET_PATH)
rows = []

VOXEL_SIZES = [0.002, 0.003, 0.004]

POISSON_DEPTHS = [7, 8, 9]

LAPLACIAN_ITERS = [2, 3, 4]

TAUBIN_ITERS = [3, 5, 7]

def load_mesh(file_name):

    mesh_path = DATA_DIR / file_name

    if not mesh_path.exists():
        print(f"{file_name} not found")
        return None

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if not mesh.has_triangles():
        print(f"{file_name} is invalid")
        return None

    mesh.compute_vertex_normals()

    return mesh

def parameter_combinations():

    for voxel in VOXEL_SIZES:
        for depth in POISSON_DEPTHS:
            for lap in LAPLACIAN_ITERS:
                for taubin in TAUBIN_ITERS:

                    yield {
    "voxel_size": voxel,
    "poisson_depth": depth,
    "laplacian_iters": lap,
    "taubin_iters": taubin
}
                    
                   
def clean_mesh_topology(mesh):

    mesh.remove_duplicated_vertices()

    mesh.remove_duplicated_triangles()

    mesh.remove_degenerate_triangles()

    mesh.remove_unreferenced_vertices()

    mesh.compute_vertex_normals()

    mesh.compute_triangle_normals()

    return mesh


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

#optimize_point_cloud()
def optimize_point_cloud(mesh, params):

    sample_count = min(
        max(len(mesh.triangles) * 3, 50000),
        200000
    )

    pcd = mesh.sample_points_uniformly(
        number_of_points=sample_count
    )

    pcd = pcd.voxel_down_sample(
        voxel_size=params["voxel_size"]
    )

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=params["voxel_size"] * 5,
            max_nn=50
        )
    )

    return pcd
              

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

def quality_score(original_mesh, optimized_mesh):

    try:
        original_area = original_mesh.get_surface_area()
        optimized_area = optimized_mesh.get_surface_area()

        area_difference = abs(original_area - optimized_area)

    except:
        area_difference = 999999

    try:
        original_triangles = len(original_mesh.triangles)
        optimized_triangles = len(optimized_mesh.triangles)

        triangle_difference = abs(
            original_triangles - optimized_triangles
        )

    except:
        triangle_difference = 999999

    score = area_difference + triangle_difference

    return score

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





rows = []

for index, row in df.iterrows():

    print("=" * 60)

    print("Loading:", row["File_Name"])

    mesh = load_mesh(row["File_Name"])

    if mesh is None:
        continue

    print("Vertices :", len(mesh.vertices))
    print("Triangles:", len(mesh.triangles))

    count = 0

    best_score = float("inf")
    best_params = None
    best_mesh = None

    for params in parameter_combinations():

        count += 1

        print(
            f"Test {count}: "
            f"Voxel={params['voxel_size']} | "
            f"Depth={params['poisson_depth']} | "
            f"Laplacian={params['laplacian_iters']} | "
            f"Taubin={params['taubin_iters']}"
        )

        try:

            optimized_mesh = optimize_mesh(
                mesh,
                params
            )

            score = quality_score(
                mesh,
                optimized_mesh
            )

            print(f"Score : {score:.4f}")

            if score < best_score:

                best_score = score
                best_params = params.copy()
                best_mesh = optimized_mesh

        except Exception as e:

            print("Optimization Failed:", e)

    # Save best parameters for this STL file
    if best_params is not None:

        rows.append({

            "File_Name": row["File_Name"],

            "Vertices": len(mesh.vertices),

            "Triangles": len(mesh.triangles),

            "Surface_Area": mesh.get_surface_area(),

            "Voxel_Size": best_params["voxel_size"],

            "Poisson_Depth": best_params["poisson_depth"],

            "Laplacian": best_params["laplacian_iters"],

            "Taubin": best_params["taubin_iters"],

            "Quality_Score": best_score

        })

        print("\nBest Parameters Found")
        print(best_params)
        print("Best Score:", best_score)

# Save training dataset
training_df = pd.DataFrame(rows)

output_csv = BASE_DIR / "dataset" / "training_dataset.csv"

training_df.to_csv(output_csv, index=False)

print("\n====================================")
print("Training Dataset Created Successfully")
print(output_csv)
print("Total Models:", len(training_df))
print("====================================")
try:

        optimized_mesh = optimize_mesh(mesh, params)

        score = quality_score(
            mesh,
            optimized_mesh
        )

        print(f"Score: {score:.4f}")

        if score < best_score:

            best_score = score

            best_params = params.copy()

            best_mesh = optimized_mesh

except Exception as e:

        print("Optimization Failed:", e)