from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from robust_exp.utils import append_csv


with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "metrics.csv"
    append_csv(path, {"epoch": 1, "clean": 0.5, "robust": 0.2})
    append_csv(path, {"epoch": 2, "robust": 0.3})
    frame = pd.read_csv(path)
    assert list(frame.columns) == ["epoch", "clean", "robust"]
    assert pd.isna(frame.loc[1, "clean"])
    try:
        append_csv(path, {"epoch": 3, "unexpected": 1})
    except ValueError as error:
        assert "unexpected fields" in str(error)
    else:
        raise AssertionError("Schema drift was not rejected")
print("CSV schema smoke test passed")
