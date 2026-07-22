from pathlib import Path

import pandas as pd

import joblib

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = BASE_DIR / "dataset" / "training_dataset.csv"

MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATASET_PATH)

print(df.head())

print("\nTotal Samples:", len(df))

# Input Features
X = df[
    [
        "Vertices",
        "Triangles",
        "Surface_Area"
    ]
]

# Target Values
y = df[
    [
        "Voxel_Size",
        "Poisson_Depth",
        "Laplacian",
        "Taubin"
    ]
]

print("\nFeatures Shape :", X.shape)
print("Targets Shape  :", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nMean Absolute Error:", mae)

model_path = MODEL_DIR / "best_model.pkl"

joblib.dump(model, model_path)

print("\nModel Saved Successfully")
print(model_path)