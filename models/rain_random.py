import random
import numpy as np

class RainRandom:
    def __init__(self):
        pass
    
    # select list of booleans according to list of probabilities
    def predict(self, X):
        """
        Converts a list of probabilities to an ndarray of binary values where 
        each value has 1 with probability X[i] and 0 with probability 1-X[i].
        :param X: list of probabilities
        :return: ndarray of binary values
        """
        
        binary_list = []
        for prob in X:
            binary_list.append(random.choices([0,1], weights=[1-prob, prob])[0])
        return np.array(binary_list)
    