from ortools.sat.python import cp_model
import pandas as pd

# -------------------------------
# DATA (change with user data)
# -------------------------------

classes = ["6A", "6B"]

days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

periods = [1, 2, 3, 4, 5, 6]

subjects = {
    "Math": {"teacher": "Mr. Sharma","hours": 5},
    "English": {"teacher": "Ms. Sen","hours": 5},
    "Science": {"teacher": "Mr. Das","hours": 4},
    "History": {"teacher": "Mr. Roy","hours": 3},
    "Computer": {"teacher": "Mrs. Gupta","hours": 3},
    "Free": {"teacher": "FREE", "hours": 10}
}

# -------------------------------
# MODEL
# -------------------------------

model = cp_model.CpModel()

# Decision variables
# x[(class, day, period, subject)] = 1 if subject is scheduled

x = {}

for c in classes:
    for d in days:
        for s in subjects:
            for p in range(1, len(periods)):
                model.Add(
                    x[(c, d, p, s)] +
                    x[(c, d, p + 1, s)]
                    <= 1
                )

for c in classes:
    for d in days:
        for p in periods:
            for s in subjects:
                x[(c, d, p, s)] = model.NewBoolVar(
                    f"{c}_{d}_{p}_{s}"
                )

# -------------------------------
# Constraint 1
# GENERATE SUBJECT SLOTS
# -------------------------------

subject_slots = []

for cls in classes:
    for subject, info in subjects.items():
        for _ in range(info["hours"]):
            subject_slots.append((cls, subject))

# -------------------------------
# Constraint 2
# Required subject hours
# -------------------------------

for c in classes:

    for s in subjects:

        model.Add(

            sum(

                x[(c, d, p, s)]

                for d in days
                for p in periods

            )

            == subjects[s]["hours"]

        )

# -------------------------------
# Constraint 3
# Teacher conflict
# -------------------------------

teachers = {}

for s in subjects:
    teacher = subjects[s]["teacher"]

    teachers.setdefault(teacher, []).append(s)

for teacher in teachers:

    teacher_subjects = teachers[teacher]

    for d in days:

        for p in periods:

            model.Add(

                sum(

                    x[(c, d, p, s)]

                    for c in classes

                    for s in teacher_subjects

                )

                <= 1

            )

# -------------------------------
# SOLVE
# -------------------------------

solver = cp_model.CpSolver()

status = solver.Solve(model)

# -------------------------------
# PRINT TIMETABLE
# -------------------------------

if status == cp_model.FEASIBLE or status == cp_model.OPTIMAL:

    print("\nTimetable Generated Successfully\n")

    rows = []

    for c in classes:

        print("=" * 40)
        print(c)
        print("=" * 40)

        for d in days:

            print("\n", d)

            for p in periods:

                for s in subjects:

                    if solver.Value(x[(c, d, p, s)]):

                        teacher = subjects[s]["teacher"]

                        print(
                            f"Period {p}: {s} ({teacher})"
                        )

                        rows.append([
                            c,
                            d,
                            p,
                            s,
                            teacher
                        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Class",
            "Day",
            "Period",
            "Subject",
            "Teacher"
        ]
    )

    df.to_csv("generated_timetable.csv", index=False)

    print("\nCSV Saved Successfully!")

else:

    if status == cp_model.OPTIMAL:
        print("Optimal timetable found")

    elif status == cp_model.FEASIBLE:
        print("Feasible timetable found")

    elif status == cp_model.INFEASIBLE:
        print("Impossible constraints")

    elif status == cp_model.MODEL_INVALID:
        print("Model Invalid")

    else:
        print(status)