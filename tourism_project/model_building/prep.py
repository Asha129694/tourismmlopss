"""Data preparation: clean, split and save train/test CSVs."""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "tourism_project/data/tourism.csv"
TARGET = "ProdTaken"

df = pd.read_csv(DATA_PATH)

# Drop unnecessary columns: the unnamed index column (if present) and the unique ID
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")] + ["CustomerID"])

# Fix inconsistent categories
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

# Handle missing values (numeric -> median, categorical -> mode) and duplicates
for col in df.columns:
    if df[col].isna().any():
        df[col] = df[col].fillna(df[col].median() if df[col].dtype.kind in "if" else df[col].mode()[0])
df = df.drop_duplicates().reset_index(drop=True)

X, y = df.drop(columns=[TARGET]), df[TARGET]
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(f"Cleaned data: {df.shape}")
print(f"Train: {Xtrain.shape}, Test: {Xtest.shape}")
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")
