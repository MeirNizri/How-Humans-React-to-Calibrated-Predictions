import os
import joblib
import pandas as pd

from sklearn.neural_network import MLPClassifier

def rain_nn():
    """
    Trains a neural network to predict whether it will rain tomorrow or not.
    """

    # If the model is already trained, load it from disk. Otherwise, train the model and save it to disk.
    model_path = os.path.join(os.path.dirname(__file__), 'rain_nn.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)

    # Reading the datasets and splitting into X and y
    train_df = pd.read_csv("datasets/cleaned_weatherAUS_train.csv")
    X_train = train_df.drop(["RainTomorrow"], axis=1)
    y_train = train_df["RainTomorrow"]

    # Initializing the NN
    model = MLPClassifier(hidden_layer_sizes=(64, 32, 16, 8),
                          activation='relu',
                          learning_rate_init=0.0001,
                          batch_size=32,
                          solver='adam',
                          max_iter=500,
                          verbose=True)

    # Training the NN
    model.fit(X_train, y_train)

    # Save the model to disk
    joblib.dump(model, model_path)

    return model
