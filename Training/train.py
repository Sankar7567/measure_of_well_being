import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add current directory to path to enable local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import preprocessing
import visualization
import model

def main():
    print("==========================================================")
    print("Human Development Index (HDI) Prediction Model Training")
    print("==========================================================\n")

    # Define absolute paths for outputs
    training_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(training_dir)
    
    dataset_dir = os.path.join(project_root, "Dataset")
    flask_static_img_dir = os.path.join(project_root, "Flask", "static", "images")
    training_static_img_dir = os.path.join(training_dir, "static", "images")
    model_pkl_path = os.path.join(project_root, "Flask", "HDI.pkl")
    
    output_dirs = [flask_static_img_dir, training_static_img_dir]

    # 1. Download dataset if missing
    print("[Step 1] Loading Dataset...")
    filepath = preprocessing.download_dataset_if_missing(dataset_dir, "HDI.csv")
    
    # 2. Preprocess data
    print("\n[Step 2] Running Data Preprocessing Pipeline...")
    X_encoded, y, label_encoder = preprocessing.preprocess_pipeline(filepath)
    
    # 3. Generate Exploratory Visualizations
    print("\n[Step 3] Generating Exploratory Visualizations...")
    visualization.generate_all_plots(X_encoded, y, output_dirs)
    
    # 4. Train Test Split (90% train, 10% test, random_state=42 as per specs)
    print("\n[Step 4] Splitting Dataset into Train/Test Sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.1, random_state=42
    )
    print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"Testing set:  X_test  {X_test.shape}, y_test  {y_test.shape}")
    
    # 5. Fit the Linear Regression model
    print("\n[Step 5] Fitting Linear Regression Model...")
    regressor = model.train_linear_regression(X_train, y_train)
    
    # 6. Evaluate model
    print("\n[Step 6] Evaluating Model Performance...")
    metrics, y_pred = model.evaluate_model(regressor, X_test, y_test)
    
    # 7. Print ground truth vs predictions
    print("\n[Step 7] Comparing Ground Truth (y_test) vs Predictions (y_pred)...")
    print(f"y_test array:\n{np.array(y_test.values)}")
    print(f"y_pred array:\n{y_pred}")
    
    comparison_df = pd.DataFrame({
        "Actual HDI": y_test.values,
        "Predicted HDI": y_pred,
        "Residual": y_test.values - y_pred
    })
    print("\nGround Truth vs Prediction comparisons (First 15 test items):")
    print(comparison_df.head(15).to_string(index=False))

    # 8. Save actual vs predicted plot
    print("\n[Step 8] Generating Model Fit Plot...")
    visualization.plot_actual_vs_predicted(y_test.values, y_pred, output_dirs)

    # 9. Save serialized model for deployment
    print("\n[Step 9] Serializing Model for Flask Web App Deployment...")
    # Retrieve raw country names list from label encoder
    countries_list = list(label_encoder.classes_)
    model.save_serialized_model(regressor, label_encoder, countries_list, model_pkl_path)
    
    print("\n==========================================================")
    print("Model Training and Deployment Packaging Complete!")
    print("==========================================================")

if __name__ == "__main__":
    main()
