"""Data preparation: clean, drop non-pre-contact columns, split, and save train/test CSVs.

Business objective: predict purchase BEFORE a customer is contacted, so only attributes that
exist before the sales interaction are kept. Missing-value imputation is NOT done here - it
lives inside the scikit-learn pipeline in train.py so statistics are learned from training data only.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

# Known only during/after the sales interaction -> excluded to avoid target leakage.
# TypeofContact is excluded too: it describes the contact itself (Self Enquiry means the
# customer already contacted the company).
POST_CONTACT = ["DurationOfPitch", "NumberOfFollowups", "PitchSatisfactionScore",
                "ProductPitched", "TypeofContact"]

df = pd.read_csv(DATA_PATH)
n_raw = len(df)

# Drop the unnamed export index and the identifier
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

# Remove only true duplicate records (same CustomerID). Rows that merely share the same
# attributes but have different CustomerIDs are different customers and are kept.
df = df.drop_duplicates(subset="CustomerID").drop(columns=["CustomerID"]).reset_index(drop=True)

# Standardise inconsistent category labels
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Keep pre-contact features only
df = df.drop(columns=POST_CONTACT)

X, y = df.drop(columns=[TARGET]), df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Raw rows: {n_raw} -> after removing duplicate CustomerIDs: {len(df)}")
print(f"Dropped post-contact features: {POST_CONTACT}")
print(f"Pre-contact features kept ({X.shape[1]}): {list(X.columns)}")
print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}  (buyer rate {ytrain.mean():.3f} / {ytest.mean():.3f})")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
