# for data manipulation
import pandas as pd
import numpy as np
import json
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/singhpayal/tourism_package_prediction_dataset/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# --- DATA CLEANING & FEATURE SELECTION ---
# Drop completely unnecessary or unique index columns if present
if 'CustomerID' in tourism_dataset.columns:
    tourism_dataset = tourism_dataset.drop(columns=['CustomerID'])
    print("Dropped unnecessary column: CustomerID")

# Fix inconsistent string categories if any (e.g., typos or mixed casing common in this dataset)
if 'Gender' in tourism_dataset.columns:
    tourism_dataset['Gender'] = tourism_dataset['Gender'].replace('Fe Male', 'Female')

# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',
    'CityTier',
    'NumberOfPersonVisiting',
    'PreferredPropertyStar',
    'NumberOfTrips',
    'NumberOfChildrenVisiting',
    'MonthlyIncome',
    'PitchSatisfactionScore',
    'NumberOfFollowups',
    'DurationOfPitch'
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'MaritalStatus',
    'Designation',
    'ProductPitched',
    'Passport',
    'OwnCar'
]

# Separate features and target
X = tourism_dataset[numeric_features + categorical_features]
y = tourism_dataset[target]

# Split the dataset into training and test sets first to prevent data leakage during imputation
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# --- IMPUTE MISSING VALUES (Data Cleaning Requirement) ---
# Learn imputation values strictly from training set, then apply to both train and test sets
for col in numeric_features:
    median_val = Xtrain[col].median()
    Xtrain[col] = Xtrain[col].fillna(median_val)
    Xtest[col] = Xtest[col].fillna(median_val)

for col in categorical_features:
    mode_val = Xtrain[col].mode()[0]
    Xtrain[col] = Xtrain[col].fillna(mode_val)
    Xtest[col] = Xtest[col].fillna(mode_val)

print("Data cleaning and missing value handling completed successfully.")

# Save data splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Save feature lists to JSON files to ensure train.py and app.py dynamic loading works
with open("numeric_features.json", "w") as f:
    json.dump(numeric_features, f)
with open("categorical_features.json", "w") as f:
    json.dump(categorical_features, f)

# List of all files to upload
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv", "numeric_features.json", "categorical_features.json"]

print("Uploading cleaned data splits and feature profiles to Hugging Face...")
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="singhpayal/tourism_package_prediction_dataset",
        repo_type="dataset",
    )
print("Data splits and feature lists uploaded to Hugging Face Dataset repository successfully.")
