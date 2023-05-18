from sklearn.linear_model import LogisticRegression


class PlattScaling:
    """
    Class for performing Platt scaling
    """

    def __init__(self, model, X, y):
        """
        Perform Platt scaling calibration on the given model.
        :param model: The model to perform Platt scaling on
        :param X: Held-out calibration data, usually the validation set
        :param y: labels of X
        """
        self.model = model

        # Get the model's predictions on the data
        uncalibrated_prob = self.model.predict(X)
        uncalibrated_prob = uncalibrated_prob.reshape(-1, 1)

        # Fit a logistic regression model to the data
        self.lr = LogisticRegression()
        self.lr.fit(uncalibrated_prob, y)


    def predict(self, X):
        """
        Return the Platt scaling model's predictions on the data.
        :param X: features to predict on 
        :return: calibrated probability for each sample in X 
        """
        uncalibrated_prob = self.model.predict(X)
        uncalibrated_prob = uncalibrated_prob.reshape(-1, 1)
        return self.lr.predict_proba(uncalibrated_prob)[:, 1]
    
