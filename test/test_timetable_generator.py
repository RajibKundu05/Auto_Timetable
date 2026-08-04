import pandas as pd

import timetable_generator as tg


def test_generate_timetable_creates_csv(tmp_path):
    output_path = tmp_path / "generated_timetable.csv"

    result = tg.generate_timetable(output_path=str(output_path))

    assert result is not None
    assert output_path.exists()

    df = pd.read_csv(output_path)
    assert not df.empty
    assert {"Class", "Day", "Period", "Subject", "Teacher"}.issubset(df.columns)
