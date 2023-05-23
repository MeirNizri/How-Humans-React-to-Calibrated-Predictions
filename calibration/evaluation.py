import numpy as np

n_bins = 10

def expected_calibration_error(y_true, y_pred, num_bins=n_bins):
    """
    Calculate the expected calibration error (ECE) of a set of predictions.
    :param y_true: True labels
    :param y_pred: Predicted probabilities
    :param num_bins: Number of bins to use when calculating ECE
    :return: Expected calibration error
    """
    ece = 0.0 

    # Create equally spaced bin edges
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    for i in range(num_bins):
        # Find indices of samples falling into the current bin
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        bin_indices = (y_pred >= bin_start) & (y_pred < bin_end)

        # Calculate the number of samples in the bin
        num_samples = np.sum(bin_indices)

        if num_samples > 0:
            # Calculate the average predicted probability and true probability in the bin
            bin_pred_prob = np.mean(y_pred[bin_indices])
            bin_true_prob = np.mean(y_true[bin_indices])

            # Calculate the absolute difference and update ECE
            bin_error = np.abs(bin_true_prob - bin_pred_prob)
            ece += (num_samples / len(y_true)) * bin_error

    return ece

def max_calibration_error(y_true, y_pred, num_bins=n_bins):
    """
    Calculate the maximum calibration error (MCE) of a set of predictions.
    :param y_true: True labels
    :param y_pred: Predicted probabilities
    :param num_bins: Number of bins to use when calculating MCE
    :return: Maximum calibration error
    """
    mce = 0.0

    # Create equally spaced bin edges
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    for i in range(num_bins):
        # Find indices of samples falling into the current bin
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        bin_indices = (y_pred >= bin_start) & (y_pred < bin_end)

        # Calculate the number of samples in the bin
        num_samples = np.sum(bin_indices)

        if num_samples > 0:
            # Calculate the average predicted probability and true probability in the bin
            bin_pred_prob = np.mean(y_pred[bin_indices])
            bin_true_prob = np.mean(y_true[bin_indices])

            # Calculate the absolute difference and update MCE if bin error is larger than current MCE
            bin_error = np.abs(bin_true_prob - bin_pred_prob)
            if bin_error > mce:
                mce = bin_error

    return mce

def overconfidence_error(y_true, y_pred, num_bins=n_bins):
    """
    Calculate the overconfidence error (OCE) of a set of predictions.
    :param y_true: True labels
    :param y_pred: Predicted probabilities
    :param num_bins: Number of bins to use when calculating OCE
    :return: Overconfidence error
    """
    oce = 0.0

    # Create equally spaced bin edges
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    for i in range(num_bins):
        # Find indices of samples falling into the current bin
        bin_start = bin_edges[i]
        bin_end = bin_edges[i+1]
        bin_indices = (y_pred >= bin_start) & (y_pred < bin_end)

        # Calculate the number of samples in the bin
        num_samples = np.sum(bin_indices)

        if num_samples > 0:
            # Calculate the average predicted probability and true probability in the bin
            bin_pred_prob = np.mean(y_pred[bin_indices])
            bin_true_prob = np.mean(y_true[bin_indices])

            # Calculate the absolute difference 
            bin_error = np.abs(bin_true_prob - bin_pred_prob)
            
            # Update OCE 
            oce += (num_samples / len(y_true)) * (bin_error * np.max(bin_error, 0))

    return oce