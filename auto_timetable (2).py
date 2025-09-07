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

#creating the model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth = 15
    )

# Data Loading (CSV file)
data = pd.read_csv('Book1.csv')
df = pd.DataFrame(data)
# print(df)

# Variables
Xt = df.drop('Teacher', axis=1)
yt = df['Teacher']
Xs = df.drop('Subject', axis=1)
ys = df['Subject']

preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), ["Day","Time","Class","Room no"])])

rf_available_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier() )
]
)

X_train , X_test , y_train , y_test = train_test_split(Xt,yt,test_size=0.2,random_state=42)

rf_available_model.fit(X_train,y_train)

y_pred = rf_available_model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

# Create model
model = cp_model.CpModel()

# 1. Keep teachers and subjects as strings
teachers = list(df['Teacher'].unique())
subjects = list(df['Subject'].unique())
days = list(df['Day'].unique())
slots = list(df['Time'].unique())
room = list(df['Room no'].unique())
# print("Teachers:", teachers)
# print("Subjects:", subjects)
# print("Days:", days)
# print("Slots:", slots)

# Allowed teacher–subject mapping
allowed = {
    "PCCCS301": ["DC", "SB3"],
    "PCCCS302": ["BM","SKC","BS"],
    "HSMC301": ["SP", "MF2"],
    "ESC301": ["VM", "AI"],
    "ESC391": ['BM,SC,SC4','NDM,SC'],
    "PCCCS392": ["TC","SC", "BM","SC", "SC4","NDM"],
    "BSC301": ["SD"],
    "PCCCS393": ['KM,RP,DD','SC4,IM','MK,DP'],
    "PCCCS391": ['DC,SG,PB','NDG,SG,PB','BB,AB2,SM2','NDG,PKM,SM'],
    "TPE301": ["FENG"],
}
allowed["OFF"] = []


# # Variables: x[teacher, subject] = 1 if teacher teaches subject
# x = {}
# x = {}
# for s, teachers in allowed.items():
#     for t in teachers:
#         for d in days:
#             for sl in slots:
#                 for r in room:
#                     x[t, s, d, sl, r] = model.NewBoolVar(f"{t}_{s}_{d}_{sl}_{r}")

# # 1. Each subject assigned to exactly one teacher
# for s in subjects:
#     model.Add(sum(x[t, s, d, sl, r] for t in allowed[s] for d in days for sl in slots for r in room) == 1)

# # 2. Block disallowed teacher-subject pairs
# for s in subjects:
#     for t in teachers:
#         if t not in allowed[s]:
#             model.Add(x[t, s] == 0)

# solver = cp_model.CpSolver()
# status = solver.Solve(model)


# new_data = pd.DataFrame([nd])
# print("Predicted availability:", rf_available_model.predict(new_data))


# weights = {}
# for t in teachers:
#     for s in subjects:
#         for d in days:
#             for sl in slots:
#                 new_data = pd.DataFrame([{
#                     "Teacher": t,
#                     "Day": d,
#                     "Slot": sl,
#                     "Subject": s
#                 }])
#                 prob = rf_available_model.predict_proba(new_data)[0][1]  # probability teacher is available
#                 weights[t, s, d, sl] = int(prob * 1000)  # scale to integer for OR-Tools

# model.Maximize(
#     sum(weights[t, s, d, sl] * x[t, s, d, sl] 
#         for t in teachers for s in subjects for d in days for sl in slots)
# )

# t, s, d, sl, r = "DC", "PCCCS301", "Monday", "9:30 - 10:25", "305"

# if solver.Value(x[t, s, d, sl, r]) == 1:
#     print(f"{t} is teaching {s} on {d} at {sl} in room {r} ✅")
# else:
#     print(f"{t} is NOT teaching {s} on {d} at {sl} in room {r} ❌")


# Example new data
new_data = {"Day": "Monday",
    "Time": "9:30 - 10:25" ,
    "Class": "CSE-B",
    "Room no": "305",
    "Subject": "PCCCS301",
    "Teacher": "DC"}

# # Checking if the teacher is available at that slot or not
# if new_data["Teacher"] not in allowed[new_data["Subject"]]:
#     print(f"❌ Invalid: {new_data['Teacher']} cannot teach {new_data['Subject']}")

# df_new = pd.DataFrame([new_data])

# # Predicted teacher for this slot
# predicted_teacher = rf_available_model.predict(df_new.drop(columns=["Teacher"]))[0]

# if predicted_teacher == new_data["Teacher"]:
#     print(f"✅ {new_data['Teacher']} is available for {new_data['Subject']} at {new_data['Time']} on {new_data['Day']}")
# else:
#     print(f"❌ {new_data['Teacher']} is NOT available, model suggests {predicted_teacher} instead")

# Suggesting the best teacher for the subject at that slot
df_new = pd.DataFrame([new_data])
X_input = df_new.drop(columns=["Teacher"])

# Step 1: predict probabilities
probs = rf_available_model.predict_proba(X_input)[0]
teachers = rf_available_model.classes_

# Step 2: filter by subject
subject = new_data["Subject"]
allowed_teachers = allowed.get(subject, [])
# prob_teacher = {t: p for t, p in zip(teachers, probs) if t in allowed_teachers}

# Pick the best teacher from allowed ones
best_teacher = None
best_prob = -1
for t, p in zip(teachers, probs):
    if t in allowed_teachers and p > best_prob:
        best_teacher = t
        best_prob = p

if best_teacher:
    print(f"✅ Suggested teacher for {subject}: {best_teacher} instead of {new_data['Teacher']}")
else:
    print(f"❌ No valid teacher found for {subject}")