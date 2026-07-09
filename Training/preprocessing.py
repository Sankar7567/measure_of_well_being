import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple

PRIMARY_URL = "https://raw.githubusercontent.com/Guided-Projects/HumanDevelopmentIndex/main/Dataset/HDI.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/augarriza/HumanDevelopmentIndex_Tableau/master/Data/HDI.csv"

def download_dataset_if_missing(dataset_dir: str = "Dataset", filename: str = "HDI.csv") -> str:
    """
    Downloads the HDI dataset if it is not present in the local directory.
    Attempts the primary repository first, then falls back to a mirrored source.
    """
    os.makedirs(dataset_dir, exist_ok=True)
    filepath = os.path.join(dataset_dir, filename)

    if os.path.exists(filepath):
        print(f"Dataset already exists at: {filepath}")
        return filepath

    print(f"Dataset missing. Attempting to download from primary source...")
    try:
        urllib.request.urlretrieve(PRIMARY_URL, filepath)
        print(f"Successfully downloaded dataset to {filepath}")
    except Exception as e:
        print(f"Primary source failed (Error: {e}). Attempting fallback source...")
        try:
            urllib.request.urlretrieve(FALLBACK_URL, filepath)
            print(f"Successfully downloaded dataset from fallback to {filepath}")
        except Exception as e_fallback:
            raise RuntimeError(f"Failed to download dataset from all sources. Fallback error: {e_fallback}")

    return filepath

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads the CSV file into a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No file found at {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"Loaded dataset successfully. Shape: {df.shape}")
    return df

def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Extracts the selected features and target from the DataFrame.
    X: [Country, Life expectancy, Mean years of schooling, Gross national income (GNI) per capita, Internet users]
       Selected via column indices [2, 5, 6, 7, 67]
    y: HDI index (Index 4)
    """
    # Verify shape is sufficient
    if df.shape[1] < 68:
        raise ValueError(f"Dataset does not contain enough columns (expected at least 82 columns, got {df.shape[1]})")

    # Column index matching:
    # 2 -> Country, 5 -> Life expectancy, 6 -> Mean years of schooling, 
    # 7 -> Gross national income (GNI) per capita, 67 -> Internet users
    X = df.iloc[:, [2, 5, 6, 7, 67]].copy()
    y = df.iloc[:, 4].copy()
    
    # Rename columns to standardized clean names
    X.columns = [
        "Country", 
        "Life expectancy", 
        "Mean years of schooling", 
        "Gross national income (GNI) per capita", 
        "Internet users"
    ]
    y.name = "HDI"
    
    return X, y

def handle_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """
    Fills null entries in selected features.
    Categorical columns (Country) are skipped, and numerical columns are imputed using their column mean.
    """
    X_imputed = X.copy()
    
    # Identify numeric columns
    numeric_cols = X_imputed.select_dtypes(include=[np.number]).columns
    
    # Fill numeric columns with their mean
    for col in numeric_cols:
        null_count = X_imputed[col].isnull().sum()
        if null_count > 0:
            mean_val = X_imputed[col].mean()
            X_imputed[col] = X_imputed[col].fillna(mean_val)
            print(f"Imputed {null_count} missing values in '{col}' with column mean: {mean_val:.4f}")
            
    return X_imputed

def encode_categorical_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """
    Encodes the 'Country' categorical feature into numerical indices.
    """
    X_encoded = X.copy()
    le = LabelEncoder()
    
    # Ensure Country values are strings and sorted to guarantee consistent encoding
    X_encoded["Country"] = X_encoded["Country"].astype(str)
    
    # Fit encoder and transform column
    X_encoded["Country"] = le.fit_transform(X_encoded["Country"])
    print(f"Encoded 'Country' column using LabelEncoder. Classes count: {len(le.classes_)}")
    
    return X_encoded, le

def preprocess_pipeline(filepath: str) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder]:
    """
    Runs the complete preprocessing pipeline from raw file path to preprocessed X, y and fitted encoder.
    """
    df = load_data(filepath)
    X, y = get_features_and_target(df)
    
    # Impute missing values
    X_clean = handle_missing_values(X)
    
    # Encode categorical country column
    X_encoded, le = encode_categorical_features(X_clean)
    
    # Ensure target y does not have missing values
    if y.isnull().any():
        print(f"Imputing missing values in target 'HDI' with column mean")
        y = y.fillna(y.mean())
        
    return X_encoded, y, le

if __name__ == "__main__":
    # Test script locally
    filepath = download_dataset_if_missing("../Dataset", "HDI.csv")
    X, y, le = preprocess_pipeline(filepath)
    print("Preprocessing completed. X head:")
    print(X.head())
    print("y head:")
    print(y.head())
