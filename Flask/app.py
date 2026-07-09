import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Locate model pickle path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "HDI.pkl")

# Global variables for model and metadata
model = None
encoder = None
countries = []
feature_names = []

def load_prediction_model():
    global model, encoder, countries, feature_names
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                data = pickle.load(f)
                model = data["model"]
                encoder = data["encoder"]
                countries = data["countries"]
                feature_names = data["feature_names"]
            print("Successfully loaded model and metadata.")
        except Exception as e:
            print(f"Error loading model pickle: {e}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}. Run training first.")

# Load model upon application startup
load_prediction_model()

@app.route("/")
@app.route("/home")
def home():
    """
    Renders the landing page explaining HDI.
    """
    return render_template("home.html")

@app.route("/Prediction", methods=["GET", "POST"])
def prediction():
    """
    Renders the prediction form, dynamically populating the country dropdown.
    """
    # Force reload of model if it failed to load on startup
    if model is None:
        load_prediction_model()

    if model is None:
        return render_template("result.html", error_msg="Prediction model is not trained/loaded yet. Please train the model first.")

    # Create list of tuples (encoded_val, country_name) for dropdown select option
    country_options = list(enumerate(countries))
    return render_template("indexnew.html", country_options=country_options)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Retrieves user inputs, executes prediction model, classifies development status,
    and displays outcome.
    """
    if model is None:
        return render_template("result.html", error_msg="Model file missing. Train model first.")

    try:
        # 1. Retrieve and validate form inputs
        form_keys = [
            "Country", 
            "Life expectancy", 
            "Mean years of schooling", 
            "Gross national income (GNI) per capita", 
            "Internet users"
        ]
        
        input_values = []
        for key in form_keys:
            val_str = request.form.get(key)
            if val_str is None or val_str.strip() == "":
                raise ValueError(f"Missing input parameter: {key}")
            input_values.append(float(val_str))
            
        # Parse individual parameters for display
        country_code = int(input_values[0])
        life_exp = input_values[1]
        mean_school = input_values[2]
        gni = input_values[3]
        internet = input_values[4]
        
        # Get country name
        if 0 <= country_code < len(countries):
            country_name = countries[country_code]
        else:
            country_name = f"Unknown (Code: {country_code})"
            
        # 2. Build input DataFrame matching the format expected by the model
        features_value = [np.array(input_values)]
        df_input = pd.DataFrame(features_value, columns=feature_names)
        
        # 3. Perform prediction
        output = model.predict(df_input)
        
        # Linear Regression prediction output could be an array or list of arrays depending on training data format
        if hasattr(output[0], "__len__") and not isinstance(output[0], (str, bytes)):
            predicted_hdi = float(output[0][0])
        else:
            predicted_hdi = float(output[0])
            
        y_pred = round(predicted_hdi, 4)
        
        # 4. Classify well-being category based on project constraints
        # Low HDI: 0.1 <= y_pred <= 0.4
        # Medium HDI: 0.4 < y_pred <= 0.7
        # High HDI: 0.7 < y_pred <= 0.8
        # Very High HDI: 0.8 < y_pred <= 0.96
        # Else: Out of range
        hdi_category = "Unknown"
        prediction_text = ""
        
        if 0.1 <= y_pred <= 0.4:
            hdi_category = "Low"
            prediction_text = f"Low HDI {y_pred:.2f}"
        elif 0.4 < y_pred <= 0.7:
            hdi_category = "Medium"
            prediction_text = f"Medium HDI {y_pred:.2f}"
        elif 0.7 < y_pred <= 0.8:
            hdi_category = "High"
            prediction_text = f"High HDI {y_pred:.2f}"
        elif 0.8 < y_pred <= 0.96:
            hdi_category = "Very High"
            prediction_text = f"Very High HDI {y_pred:.2f}"
        else:
            hdi_category = "Out of Range"
            prediction_text = "The given values do not match the range of values"

        return render_template(
            "result.html",
            prediction_text=prediction_text,
            hdi_value=y_pred,
            hdi_category=hdi_category,
            country_name=country_name,
            life_expectancy=life_exp,
            mean_schooling=mean_school,
            gni_per_capita=gni,
            internet_users=internet
        )

    except Exception as e:
        return render_template("result.html", error_msg=f"Error executing prediction: {e}")

@app.route("/predict", methods=["GET"])
def predict_get():
    """
    Redirect GET requests on predict route to the prediction form.
    """
    return redirect(url_for("prediction"))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
