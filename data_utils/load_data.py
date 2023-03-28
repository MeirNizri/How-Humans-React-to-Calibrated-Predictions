import numpy as np
import pandas as pd
from models import RainNN, RainRandom


def get_model_prediction(model_name="rain_nn", num_samples=20):
    """
    Get model prediction and true label for given number of samples.
    :param model_name: name of the model to use
    :param num_samples: number of samples to predict
    :return: y_pred, y - model prediction and true label
    """

    # if model is "random" sample random probabilities and get their prediction from RainRandom
    if model_name == "random":
        rain_random = RainRandom()
        y_pred = np.random.random(num_samples)
        y = rain_random.predict(y_pred)

    # if model is "rain_nn" sample random samples from the dataset and make predictions
    elif model_name == "rain_nn":
        # Load the data and instantiate the models
        rain_nn = RainNN()
        weather_df = pd.read_csv("dataset\cleaned_weatherAUS_test.csv")
        
        # Sample random samples from the dataset and make predictions
        days_to_predict = weather_df.sample(num_samples)
        X = days_to_predict.drop(["RainTomorrow"], axis=1)
        y = days_to_predict["RainTomorrow"]
        y_pred = rain_nn.predict(X).reshape(-1)

    return y_pred, y
