from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Load dataset
# -----------------------------

df = pd.read_csv("timetable_school.csv")

# -----------------------------
# Features
# -----------------------------

X = df[[
    "Day",
    "Time",
    "Class",
    "Section",
    "Room no",
    "Subject"
]]

# Target
y = df["Teacher"]

# -----------------------------
# Preprocessing
# -----------------------------

categorical_features = [
    "Day",
    "Time",
    "Class",
    "Section",
    "Room no",
    "Subject"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# -----------------------------
# Random Forest
# -----------------------------

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        random_state=42,
        class_weight="balanced"
    ))
])

# -----------------------------
# Train/Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Train
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------

predictions = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, predictions))

print()

print(classification_report(
    y_test,
    predictions
))

# -----------------------------
# Save
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predict_teacher_model.pkl"

joblib.dump(model, MODEL_PATH)

print("\nModel Saved Successfully!")