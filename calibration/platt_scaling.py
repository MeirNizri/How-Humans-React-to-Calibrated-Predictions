from sklearn.calibration import CalibratedClassifierCV


def platt_scaling(model, X, y):
    """
    Perform Platt scaling calibration on the given model.
    :param model: The model to perform Platt scaling on
    :param X: Held-out calibration data, usually the validation set
    :param y: labels of X

    Returns:
        object: Calibrated model with Platt scaling.
    """

    # Fit a logistic regression model to the data
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv="prefit")
    calibrated_model.fit(X, y)

    return calibrated_model
