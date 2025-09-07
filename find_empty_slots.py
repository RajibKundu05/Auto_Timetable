import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from ortools.sat.python import cp_model
import itertools

# Data Loading (CSV file)
data = pd.read_csv('Book1.csv')
df = pd.DataFrame(data)

teachers = list(df['Teacher'].unique())
subjects = list(df['Subject'].unique())
days = list(df['Day'].unique())
slots = list(df['Time'].unique())
room = list(df['Room no'].unique())

# Generate all possible slots (cartesian product)
all_slots = pd.DataFrame(
    list(itertools.product(days, slots, subjects, room)),
    columns=["Day", "Time", "Class", "Room no"]
)

# Take only columns relevant for slot matching
occupied_slots = df[["Day", "Time", "Class", "Room no"]]

# Find slots that are OFF or have no subject
empty_slots = (
    df.groupby(["Day", "Time","Room no", "Class"])["Subject"]
      .apply(lambda subs: all(s == "OFF" for s in subs))  # True if all OFF
      .reset_index()
)

# Keep only the empty ones
empty_slots = empty_slots[empty_slots["Subject"] == True].drop(columns=["Subject"])

print("Empty slots (no classes or OFF):")
print(len(empty_slots))
print(empty_slots)
