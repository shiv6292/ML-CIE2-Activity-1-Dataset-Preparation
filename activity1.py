
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("healthcare_dataset.csv")

print("\nDataset Loaded Successfully!")

print("\nDataset Name: Healthcare Dataset")
print("File Type: CSV")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst Five Records:")
print(df.head())

print("\nLast Five Records:")
print(df.tail())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

print("\nNumerical Columns:")
print(list(num_cols))

print("\nCategorical Columns:")
print(list(cat_cols))

print("\nStatistical Summary:")
print(df.describe())

print("\nUnique Values:")
for col in cat_cols:
    print("\n", col)
    print(df[col].unique())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nMissing Value Percentage:")
print((df.isnull().sum() / len(df)) * 100)

print("\nColumns Without Missing Values:")
print(df.columns[df.isnull().sum() == 0].tolist())

df_clean = df.copy()

df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

df_clean["Medical Condition"] = df_clean["Medical Condition"].fillna(
    df_clean["Medical Condition"].mode()[0]
)

print("\nRemaining Missing Values:")
print(df_clean.isnull().sum())

print("\nTotal Duplicate Rows:", df_clean.duplicated().sum())

print("\nDuplicate Records:")
print(df_clean[df_clean.duplicated()])

df_clean = df_clean.drop_duplicates()

print("\nRemaining Duplicates:", df_clean.duplicated().sum())

for col in df_clean.select_dtypes(include="object").columns:
    df_clean[col] = df_clean[col].str.strip()

print("\nExtra Spaces Removed Successfully!")

print("\nCleaned Dataset Summary:")
print(df_clean.describe())

print("\nMedical Condition Frequency:")
print(df_clean["Medical Condition"].value_counts())

plt.figure(figsize=(8, 5))
plt.hist(df_clean["Age"], bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 5))
df_clean["Medical Condition"].value_counts().plot(kind="bar")
plt.title("Medical Conditions Distribution")
plt.xlabel("Medical Condition")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

categorical_cols = [
    "Gender",
    "Blood Type",
    "Medical Condition",
    "Insurance Provider",
    "Admission Type",
    "Medication",
    "Test Results"
]

print("\nCategorical Columns:")
print(categorical_cols)

print("\nBefore Encoding:")
print(df_clean[categorical_cols].head())

df_model = df_clean.drop(
    columns=[
        "Name",
        "Doctor",
        "Hospital",
        "Room Number",
        "Date of Admission",
        "Discharge Date"
    ]
)

print("\nUnnecessary Columns Removed!")
print(df_model.head())

df_encoded = pd.get_dummies(
    df_model,
    columns=categorical_cols,
    dtype=int
)

print("\nOne-Hot Encoding Applied Successfully!")
print(df_encoded.head())

print("\nShape Before Encoding:", df_model.shape)
print("Shape After Encoding:", df_encoded.shape)

scaler = StandardScaler()

numeric_cols = ["Age", "Billing Amount"]

df_scaled = df_encoded.copy()

df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

print("\nStandard Scaling Applied Successfully!")

print("\nBefore Scaling:")
print(df_encoded[numeric_cols].head())

print("\nAfter Scaling:")
print(df_scaled[numeric_cols].head())

print("\nFinal Dataset Shape:", df_scaled.shape)

print("\nTotal Missing Values:", df_scaled.isnull().sum().sum())

print("Total Duplicate Rows:", df_scaled.duplicated().sum())

df_scaled.to_csv("healthcare_prepared.csv", index=False)

print("\nPrepared Dataset Saved Successfully!")

if os.path.exists("healthcare_prepared.csv"):
    print("File Created Successfully!")
    print("File Size:", os.path.getsize("healthcare_prepared.csv"), "bytes")

print("\nFinal Prepared Dataset:")
print(df_scaled.head())

print("\nACTIVITY 1 COMPLETED SUCCESSFULLY!")
