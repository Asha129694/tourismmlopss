"""Automated data-quality tests, run by the workflow before any training."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_building"))
import data_register  # noqa: E402

DF = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "tourism.csv")


def test_expected_columns_present():
    assert set(data_register.EXPECTED_COLUMNS) <= set(DF.columns)


def test_dataset_passes_validation():
    assert data_register.validate(DF) == []


def test_target_is_binary_and_has_both_classes():
    assert set(DF["ProdTaken"].unique()) == {0, 1}


def test_enough_rows_and_class_balance():
    assert len(DF) >= 1000
    assert 0.05 <= DF["ProdTaken"].mean() <= 0.60


def test_customer_ids_unique():
    assert DF["CustomerID"].is_unique
