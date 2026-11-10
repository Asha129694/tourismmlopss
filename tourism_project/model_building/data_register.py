"""Data registration: validate the raw dataset that lives inside the repository.

The raw file holds the 20 business columns plus an unnamed index column that comes from
the original export (it has gaps, so it is not a record count). It is dropped in prep.py.
"""
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]

# Valid ranges / value sets used by the quality checks
BINARY_COLUMNS = ["ProdTaken", "Passport", "OwnCar"]
NUMERIC_RANGES = {
    "Age": (18, 100), "CityTier": (1, 3), "PreferredPropertyStar": (3, 5),
    "PitchSatisfactionScore": (1, 5), "NumberOfPersonVisiting": (1, 10),
    "NumberOfFollowups": (0, 10), "NumberOfTrips": (0, 50),
    "NumberOfChildrenVisiting": (0, 5), "DurationOfPitch": (1, 300),
    "MonthlyIncome": (1, 1_000_000),
}


def validate(df):
    """Return a list of data-quality problems (empty list = dataset is valid)."""
    errors = []
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for col in BINARY_COLUMNS:
        bad = set(df[col].dropna().unique()) - {0, 1}
        if bad:
            errors.append(f"{col} has non-binary values: {sorted(bad)}")
    if df["CustomerID"].duplicated().any():
        errors.append(f"{int(df['CustomerID'].duplicated().sum())} duplicated CustomerID values")
    for col, (lo, hi) in NUMERIC_RANGES.items():
        out = df[col].dropna()
        out = out[(out < lo) | (out > hi)]
        if len(out):
            errors.append(f"{col} has {len(out)} values outside [{lo}, {hi}]")
    if (df["NumberOfChildrenVisiting"] > df["NumberOfPersonVisiting"]).any():
        errors.append("NumberOfChildrenVisiting exceeds NumberOfPersonVisiting")
    return errors


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    problems = validate(df)
    if problems:
        sys.exit("Validation failed:\n- " + "\n- ".join(problems))

    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    print("Dataset registered successfully - all validation checks passed")
    print(f"Path            : {DATA_PATH}")
    print(f"Shape           : {df.shape[0]} rows x {df.shape[1]} columns "
          f"({len(EXPECTED_COLUMNS)} business columns + extra: {extra})")
    print(f"Target balance  : {df['ProdTaken'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"Missing values  : {int(df.isna().sum().sum())}")
    print(f"Unique customers: {df['CustomerID'].nunique()}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
