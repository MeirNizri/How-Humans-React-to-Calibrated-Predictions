import numpy as np
import matplotlib.pyplot as plt

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
    
    # plot probability weighting function and inverse probability weighting function
    p = np.arange(0, 1.001, 0.001)
    plt.plot(p, probability_weighting(p), color='red', label='weighting function')
    plt.plot(p, inverse_probability_weighting(p), color='blue', label='inverse weighting function')
    plt.plot(p, p, '--', color='black', label='w(p)=p')
    plt.xlabel('p')
    plt.ylabel('w(p)')
    plt.legend()
    plt.show()
