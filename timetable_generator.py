from __future__ import annotations

from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

# -------------------------------
# DATA (change with user data)
# -------------------------------

classes = ["6A", "6B"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
periods = [1, 2, 3, 4, 5, 6]

subjects = {
    "Math": {"teacher": "Mr. Sharma", "hours": 5},
    "English": {"teacher": "Ms. Sen", "hours": 5},
    "Science": {"teacher": "Mr. Das", "hours": 4},
    "History": {"teacher": "Mr. Roy", "hours": 3},
    "Computer": {"teacher": "Mrs. Gupta", "hours": 3},
    "Free": {"teacher": "FREE", "hours": 10},
}


def generate_timetable(output_path: str | None = None) -> pd.DataFrame:
    """Create a simple feasible timetable and save it to a CSV file."""
    model = cp_model.CpModel()

    # x[(class, day, period, subject)] = 1 if the subject is scheduled at that slot
    x = {}
    for cls in classes:
        for day in days:
            for period in periods:
                for subject in subjects:
                    x[(cls, day, period, subject)] = model.NewBoolVar(
                        f"{cls}_{day}_{period}_{subject}"
                    )

    # One subject per class, per day, per period
    for cls in classes:
        for day in days:
            for period in periods:
                model.Add(
                    sum(x[(cls, day, period, subject)] for subject in subjects) == 1
                )

    # Each subject must appear for the required number of hours for each class
    for cls in classes:
        for subject, info in subjects.items():
            model.Add(
                sum(
                    x[(cls, day, period, subject)]
                    for day in days
                    for period in periods
                )
                == info["hours"]
            )

    # Avoid assigning the same subject in consecutive periods on the same day
    for cls in classes:
        for day in days:
            for period in periods[:-1]:
                for subject in subjects:
                    model.Add(
                        x[(cls, day, period, subject)]
                        + x[(cls, day, period + 1, subject)]
                        <= 1
                    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Unable to build a timetable. Solver status: {status}")

    rows = []
    for cls in classes:
        for day in days:
            for period in periods:
                for subject in subjects:
                    if solver.Value(x[(cls, day, period, subject)]):
                        teacher = subjects[subject]["teacher"]
                        rows.append([cls, day, period, subject, teacher])

    df = pd.DataFrame(
        rows,
        columns=["Class", "Day", "Period", "Subject", "Teacher"],
    )

    output_file = Path(output_path or "generated_timetable.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    timetable = generate_timetable()
    print("\nTimetable Generated Successfully\n")
    print(timetable.to_string(index=False))
    print("\nCSV Saved Successfully!")