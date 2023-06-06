import numpy as np
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer

def temperature_scaling(model, X, y, temperature_range=(0.1, 10.0), num_steps=100):
    """
    Performs temperature scaling on a probabilistic model.
    :model: The model to perform Platt scaling on
    :param X: Held-out calibration data, usually the validation set
    :param y: labels of X
    :temperature_range: Range of temperature values to search for the best parameter T.
    :num_steps: Number of steps to search for the best parameter T.

    Returns:
        object: Calibrated model with temperature scaling.
    """

    # Convert the labels to one-hot encoded vectors
    y_one_hot = LabelBinarizer().fit_transform(y)

    # Initialize variables to track the best T and its corresponding loss
    best_loss = np.inf
    best_T = None

    # Search for the best T within the specified range
    T_values = np.linspace(temperature_range[0], temperature_range[1], num_steps)
    for T in T_values:
        # Apply temperature scaling to the model's logits
        logits = model.predict_log_proba(X) / T

        # Calculate the loss using the negative log-likelihood (cross-entropy)
        loss = log_loss(y_one_hot, np.exp(logits))

        # Save the best T and its corresponding loss
        if loss < best_loss:
            best_loss = loss
            best_T = T

    # Apply temperature scaling with the best T to the model's probabilities
    calibrated_model = model
    calibrated_model.predict_proba = lambda X: np.exp(model.predict_log_proba(X) / best_T)

    print("Best Temperature: {:.3f} | Best Loss: {:.4f}".format(best_T, best_loss))
    return calibrated_model
