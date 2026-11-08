"""Data registration: validate the raw dataset that lives inside the repository."""
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

df = pd.read_csv(DATA_PATH)

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    sys.exit(f"Validation failed - missing columns: {missing}")

print("Dataset registered successfully")
print(f"Path          : {DATA_PATH}")
print(f"Shape         : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Target balance: {df['ProdTaken'].value_counts(normalize=True).round(3).to_dict()}")
print(f"Missing values: {int(df.isna().sum().sum())}")
print("\nColumn dtypes:")
print(df.dtypes.to_string())
