import numpy as np
from calibration import platt_scaling


class OnlinePlattScaling:
    """
    Class for performing Online Platt Scaling with Calibeating (OPS).
    """

    def __init__(self, model, X, y, num_bins=10):
        """
        Perform Online Platt Scaling with Calibeating on the given model.
        :param model: The model to calibrate
        :param X: Held-out calibration data, usually the validation set
        :param y: labels of X
        """

        # Perform Platt scaling on the given model
        self.ps_model = platt_scaling(model, X, y)
        cal_probs = self.ps_model.predict_proba(X)[:, 1]

        # Create bins for calibeating
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        self.bin_avg = np.zeros(num_bins)

        # Calculate the average of true predictions in each bin
        for i in range(num_bins):
            # Find indices within the current bin range
            bin_start = self.bin_edges[i]
            bin_end = self.bin_edges[i+1]
            bin_indices = (cal_probs >= bin_start) & (cal_probs < bin_end)

            # Calculate the number of samples in the bin
            num_samples = np.sum(bin_indices)

            if num_samples > 0:
                # Calculate the average of true predictions in the bin
                self.bin_avg[i] = np.mean(y[bin_indices])
            else:
                self.bin_avg[i] = (bin_start + bin_end) / 2

    def predict_proba(self, X):
        """
        Return the OPS predictions on the data.
        :param X: features to predict on 
        :return: calibrated probability for each sample in X 
        """
        # Get the model's predictions on the data
        ps_probs = self.ps_model.predict_proba(X)[:, 1]
        ps_probs = ps_probs.reshape(-1, 1) 

        # Calibeate the predictions
        cal_probs = np.zeros((len(ps_probs),2))
        for i in range(len(ps_probs)):
            # Find the bin that the prediction falls into
            bin_index = np.digitize(ps_probs[i], self.bin_edges) - 1

            # Set the calibrated prediction to the average of the bin
            cal_probs[i] = [1-self.bin_avg[bin_index], self.bin_avg[bin_index]]

        return cal_probs
