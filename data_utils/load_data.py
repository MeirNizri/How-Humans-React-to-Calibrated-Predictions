import random
import numpy as np
import pandas as pd

from calibration import inverse_probability_weighting
from models import get_rain_nn, calibrate_rain_nn


# Load the test data
weather_df = pd.read_csv("datasets/cleaned_weatherAUS_test.csv")


def get_model_prediction(model_name, num_samples=20):
    """
    Get model prediction and true label for given number of samples.
    :param model_name: name of the model to use
    :param num_samples: number of samples to predict
    :return: y_pred, y - model prediction and true label
    """
    if model_name == "perfect(uniform)":
        # Sample random probabilities 
        y_pred = np.random.random(num_samples)

        # Choose binary values according to the probabilities
        binary_list = []
        for prob in y_pred:
            binary_list.append(random.choices([0,1], weights=[1-prob, prob])[0])
        y = np.array(binary_list)
    
    elif model_name == "ignorant(high-low)":
        # Sample random probabilities between 0-0.25 and 0.75-1
        y_pred = np.random.uniform(0, 0.5, size=num_samples)
        y_pred[y_pred > 0.25] += 0.5

        # Choose 0 or 1 randomly
        y = np.random.randint(0, 2, size=num_samples)

    elif model_name == "pt_calibrated(uniform)":
        # Sample random probabilities and add prospect theory weighting
        random_pred = np.random.random(num_samples)
        y_pred = inverse_probability_weighting(random_pred)

        # Choose binary values according to the probabilities
        binary_list = []
        for prob in random_pred:
            binary_list.append(random.choices([0,1], weights=[1-prob, prob])[0])
        y = np.array(binary_list)

    elif model_name == "NN":
        # Create nn model and get model prediction and true label for given number of samples
        rain_nn = get_rain_nn()
        X, y = get_samples(num_samples)
        y_pred = rain_nn.predict_proba(X)[:, 1].reshape(-1)
    
    elif model_name == "NN_PT":
        # Create nn model and get model prediction and true label for given number of samples
        rain_nn = get_rain_nn()
        X, y = get_samples(num_samples)
        y_pred = rain_nn.predict_proba(X)[:, 1].reshape(-1)
 
        # add prospect theory weighting
        y_pred = inverse_probability_weighting(y_pred)

    elif model_name == "NN_IR":
        # Create calibrated model
        rain_nn = get_rain_nn()
        calibrated_rain_nn = calibrate_rain_nn(rain_nn, "ir")

        # Get model prediction and true label for given number of samples
        X, y = get_samples(num_samples)
        y_pred = calibrated_rain_nn.predict_proba(X)[:, 1].reshape(-1)

    elif model_name == "NN_IR_PT":
        # Create nn model and get model prediction and true label for given number of samples
        rain_nn = get_rain_nn()
        calibrated_rain_nn = calibrate_rain_nn(rain_nn, "ir")

        X, y = get_samples(num_samples)
        y_pred = calibrated_rain_nn.predict_proba(X)[:, 1].reshape(-1)
 
        # add prospect theory weighting
        y_pred = inverse_probability_weighting(y_pred)

    elif model_name == "NN_IR_PT_Random":
        # Create nn model and get model prediction and true label for given number of samples
        rain_nn = get_rain_nn()
        calibrated_rain_nn = calibrate_rain_nn(rain_nn, "ir")

        X, _ = get_samples(num_samples)
        y_pred = calibrated_rain_nn.predict_proba(X)[:, 1].reshape(-1)
 
        # add prospect theory weighting
        y_pred = inverse_probability_weighting(y_pred)

        # create series of randomly cosen 0s and 1s in the same shape as y_pred
        y = pd.Series(np.random.randint(0, 2, size=num_samples))

    return y_pred, y


def get_samples(num_samples=20):
    """
    Get random samples from the dataset.
    :param num_samples: number of samples to predict
    """
    days_to_predict = weather_df.sample(num_samples)
    X = days_to_predict.drop(["RainTomorrow"], axis=1)
    y = days_to_predict["RainTomorrow"]

    return X, y
