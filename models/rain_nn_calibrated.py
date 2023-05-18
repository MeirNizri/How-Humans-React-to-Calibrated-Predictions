import os
import pickle
import pandas as pd
from calibration import PlattScaling, OnlinePlattScaling


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
        with open(model_path, "rb") as file:
            calibrated_model = pickle.load(file)
        return calibrated_model
    
    # Load data
    weather_df = pd.read_csv("datasets/cleaned_weatherAUS_validation.csv")
    X = weather_df.drop(["RainTomorrow"], axis=1)
    y = weather_df["RainTomorrow"]

    if method == "ops":
        # Perform online platt scaling
        calibrated_model = OnlinePlattScaling(model, X, y)

    elif method == "ps":
        # Perform platt scaling
        calibrated_model = PlattScaling(model, X, y)

    # Save calibrated model
    with open(model_path, "wb") as file:
        pickle.dump(calibrated_model, file)
    
    return calibrated_model 
