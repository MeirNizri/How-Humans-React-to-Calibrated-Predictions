from sklearn.calibration import CalibratedClassifierCV


def isotonic_regression(model, X, y):
    """
    Perform Isotonic Regression calibration on the given model.
    :param model: The model to perform Platt scaling on
    :param X: Held-out calibration data, usually the validation set
    :param y: labels of X
    """

    # Fit a logistic regression model to the data
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv="prefit")
    calibrated_model.fit(X, y)

    return calibrated_model
