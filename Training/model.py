import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Dict, Any

def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Instantiates and trains a Scikit-learn Linear Regression model on training features.
    """
    print("Training Linear Regression model...")
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    print("Linear Regression model successfully trained.")
    return regressor

def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Evaluates model predictions on test data using R², MSE, and RMSE.
    """
    print("Evaluating model performance on test set...")
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        "r2_score": float(r2),
        "mean_squared_error": float(mse),
        "root_mean_squared_error": float(rmse),
        "intercept": float(model.intercept_)
    }
    
    print("\n--- Model Evaluation Results ---")
    print(f"R² (Coefficient of Determination): {r2:.6f}")
    print(f"Mean Squared Error (MSE):       {mse:.6f}")
    print(f"Root Mean Squared Error (RMSE):  {rmse:.6f}")
    print(f"Intercept:                      {model.intercept_:.6f}")
    
    # Show coefficients
    print("\nModel Coefficients:")
    for col, coef in zip(X_test.columns, model.coef_):
        print(f"  {col:40s}: {coef:+.6f}")
        
    return metrics, y_pred

def save_serialized_model(model: LinearRegression, encoder: LabelEncoder, countries: list, filepath: str) -> None:
    """
    Serializes and saves the trained model, label encoder, and list of country names
    into a pickle file for deployment inside the Flask app.
    """
    data = {
        "model": model,
        "encoder": encoder,
        "countries": sorted(countries),
        "feature_names": [
            "Country", 
            "Life expectancy", 
            "Mean years of schooling", 
            "Gross national income (GNI) per capita", 
            "Internet users"
        ]
    }
    
    with open(filepath, "wb") as f:
        pickle.dump(data, f)
        
    print(f"\nModel and metadata successfully serialized and saved to: {filepath}")
