import os
import joblib
import pandas as pd
from calibration import platt_scaling, isotonic_regression, OnlinePlattScaling, temperature_scaling


def calibrate_rain_nn(model, method):
    """
    Calibrate the given model using the given method.
    :param model: The model to calibrate
    :param method: The calibration method to use
    :return: The calibrated model
    """

    # If calibrated model already exists, load and return it
    model_path = os.path.join(os.path.dirname(__file__), 'rain_nn_'+method+'.pkl')
    if os.path.exists(model_path):
        with open(model_path, "rb") as path:
            calibrated_model = joblib.load(path)
        return calibrated_model
    
    # Load data
    weather_df = pd.read_csv("datasets/cleaned_weatherAUS_validation.csv")
    X = weather_df.drop(["RainTomorrow"], axis=1)
    y = weather_df["RainTomorrow"]

    # Calibrate model
    if method == "ps":
        # Perform platt scaling
        calibrated_model = platt_scaling(model, X, y)
    elif method == "ir":
        # Perform isotonic regression
        calibrated_model = isotonic_regression(model, X, y)
    elif method == "ops":
        # Perform online platt scaling
        calibrated_model = OnlinePlattScaling(model, X, y)
    elif method == "ts":
        # Perform temperature scaling
        calibrated_model = temperature_scaling(model, X, y)

    # Save calibrated model
    with open(model_path, "wb") as path:
        joblib.dump(calibrated_model, path)
    
    return calibrated_model 
