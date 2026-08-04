import joblib
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from sklearn.model_selection import train_test_split

# ====================================================
# LOAD DATASET
# ====================================================

df = pd.read_csv("timetable_school.csv")

# Features
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

# ====================================================
# LOAD TRAINED MODEL
# ====================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "predict_teacher_model.pkl"
model = joblib.load(MODEL_PATH)

print("=" * 50)
print("Model Loaded Successfully")
print("=" * 50)

# ====================================================
# TEST DATA
# ====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ====================================================
# PREDICTION
# ====================================================

y_pred = model.predict(X_test)

# ====================================================
# ACCURACY
# ====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy")

print(f"{accuracy*100:.2f}%")

# ====================================================
# CLASSIFICATION REPORT
# ====================================================

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ====================================================
# CONFUSION MATRIX
# ====================================================

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))

# ====================================================
# SHOW SOME TEST PREDICTIONS
# ====================================================

print("\nSample Predictions\n")

sample = X_test.head(10)

predictions = model.predict(sample)

for i in range(len(sample)):

    print("-----------------------------------")

    print(sample.iloc[i].to_dict())

    print("Actual Teacher    :", y_test.iloc[i])

    print("Predicted Teacher :", predictions[i])

# ====================================================
# TEST A NEW SLOT
# ====================================================

print("\n")
print("=" * 50)
print("NEW SLOT TEST")
print("=" * 50)

new_data = pd.DataFrame([{
    "Day": "Monday",
    "Time": "09:00-10:00",
    "Class": 8,
    "Section": "A",
    "Room no": "101",
    "Subject": "Mathematics"
}])

prediction = model.predict(new_data)[0]

print("\nPredicted Teacher")

print(prediction)

# ====================================================
# PROBABILITIES
# ====================================================

print("\nTeacher Probabilities\n")

probabilities = model.predict_proba(new_data)[0]

teachers = model.classes_

ranking = sorted(
    zip(teachers, probabilities),
    key=lambda x: x[1],
    reverse=True
)

for teacher, probability in ranking:

    print(f"{teacher:20} {probability*100:.2f}%")

# ====================================================
# TOP 3
# ====================================================

print("\nTop 3 Suggested Teachers\n")

for teacher, probability in ranking[:3]:

    print(f"{teacher:20} {probability*100:.2f}%")

# ====================================================
# MODEL INFORMATION
# ====================================================

print("\n")
print("=" * 50)
print("MODEL INFORMATION")
print("=" * 50)

print("Teachers Learned:")

for teacher in model.classes_:
    print("-", teacher)

print("\nTotal Teachers :", len(model.classes_))

print("Total Dataset Rows :", len(df))

print("Testing Rows :", len(X_test))

print("Training Rows :", len(X_train))

# ====================================================
# RANDOM TESTS
# ====================================================

print("\n")
print("=" * 50)
print("RANDOM TEST CASES")
print("=" * 50)

random_samples = X_test.sample(5, random_state=42)

preds = model.predict(random_samples)

for i in range(len(random_samples)):

    print("\nCase", i + 1)

    print(random_samples.iloc[i].to_dict())

    print("Prediction:", preds[i])

print("\n")
print("=" * 50)
print("MODEL TEST COMPLETED")
print("=" * 50)