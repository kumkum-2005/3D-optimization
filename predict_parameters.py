from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model Loaded Successfully")

vertices = int(input("Vertices: "))
triangles = int(input("Triangles: "))
surface_area = float(input("Surface Area: "))

sample = pd.DataFrame([{
    "Vertices": vertices,
    "Triangles": triangles,
    "Surface_Area": surface_area
}])

prediction = model.predict(sample)

print("\nRecommended Parameters")

print("Voxel Size      :", prediction[0][0])
print("Poisson Depth   :", round(prediction[0][1]))
print("Laplacian Iters :", round(prediction[0][2]))
print("Taubin Iters    :", round(prediction[0][3]))