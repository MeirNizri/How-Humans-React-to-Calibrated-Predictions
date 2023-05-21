import numpy as np
from calibration import PlattScaling


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
        self.ps_model = PlattScaling(model, X, y)
        cal_probs = self.ps_model.predict(X)

        # Create bins for calibeating
        self.bin_edges = np.linspace(0, 1, num_bins + 1)
        self.bin_avg = np.zeros(num_bins)

        # Calculate the average of true predictions in each bin
        for i in range(num_bins):
            # Find indices within the current bin range
            bin_start = self.bin_edges[i]
            bin_end = self.bin_edges[i+1]
            bin_indices = (cal_probs >= bin_start) & (cal_probs < bin_end)

            # Calculate the average of true predictions in the bin
            bin_values = y[bin_indices]
            if len(bin_values) != 0:
                self.bin_avg[i] = np.mean(bin_values)
            else:
                self.bin_avg[i] = (bin_start + bin_end) / 2

    def predict(self, X):
        """
        Return the OPS predictions on the data.
        :param X: features to predict on 
        :return: calibrated probability for each sample in X 
        """
        # Get the model's predictions on the data
        ps_probs = self.ps_model.predict(X)
        ps_probs = ps_probs.reshape(-1, 1) 

        # Calibeate the predictions
        cal_probs = np.zeros(len(ps_probs))
        for i in range(len(ps_probs)):
            # Find the bin that the prediction falls into
            bin_index = np.digitize(ps_probs[i], self.bin_edges) - 1

            # Set the calibrated prediction to the average of the bin
            cal_probs[i] = self.bin_avg[bin_index]
        return cal_probs
