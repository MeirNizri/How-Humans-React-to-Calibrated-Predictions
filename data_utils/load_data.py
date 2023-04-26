import numpy as np
import random
# from models import RainNN


def get_model_prediction(model_name="rain_nn", num_samples=20):
    """
    Get model prediction and true label for given number of samples.
    :param model_name: name of the model to use
    :param num_samples: number of samples to predict
    :return: y_pred, y - model prediction and true label
    """

    if model_name == "random":
        # sample random probabilities 
        y_pred = np.random.random(num_samples)

        # choose binary values according to the probabilities
        binary_list = []
        for prob in y_pred:
            binary_list.append(random.choices([0,1], weights=[1-prob, prob])[0])
        y = np.array(binary_list)
    

    elif model_name == "overconfident":
        # sample random probabilities between 0-0.25 and 0.75-1
        y_pred = np.random.uniform(0, 0.5, size=num_samples)
        y_pred[y_pred > 0.25] += 0.5

        # choose 0 or 1 randomly
        y = np.random.randint(0, 2, size=num_samples)

        
    # elif model_name == "rain_nn":
    #     # Load the data and instantiate the models
    #     rain_nn = RainNN()
    #     weather_df = pd.read_csv("dataset\cleaned_weatherAUS_test.csv")
        
    #     # Sample random samples from the dataset and make predictions
    #     days_to_predict = weather_df.sample(num_samples)
    #     X = days_to_predict.drop(["RainTomorrow"], axis=1)
    #     y = days_to_predict["RainTomorrow"]
    #     y_pred = rain_nn.predict(X).reshape(-1)


    return y_pred, y
