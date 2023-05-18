import numpy as np

def probability_weighting(p, alpha=0.71):
    """
    Probability weighting function according to prospect theory.
    :param p: probability
    :param alpha: alpha parameter
    :return: probability weighting of p
    """
    return (np.power(p, alpha) / np.power((np.power(p, alpha) + np.power(1-p, alpha)), 1/alpha))


def inverse_probability_weighting(p, alpha=0.71):
    """
    Inverse probability weighting function according to prospect theory.
    :param p: probability
    :param alpha: alpha parameter
    :return: inverse probability weighting of p 
    """
    weighted_probabilities = (np.power(p, 1/alpha) / np.power((np.power(p, 1/alpha) + np.power(1-p, 1/alpha)), 1/alpha))
    return np.minimum(weighted_probabilities, 1)

if __name__ == '__main__':
    # Test inverse probability weighting functions for different values of p 
    for p in np.arange(0, 1.1, 0.1):
        # p_hat = probability_weighting(p)
        p_hat = inverse_probability_weighting(p)

        # p_hat should be equal to p 
        print(f"{p:.1f} -> {p_hat:.3f}")
    
    # random_pred = np.random.random(20)
    # print(type(random_pred))
    # print(type(inverse_probability_weighting(random_pred)))


