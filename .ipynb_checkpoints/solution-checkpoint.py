from sklearn.datasets import load_digits
import numpy as np


digits = load_digits()
flat_image = np.array(digits.data[0]).reshape(digits.data[0].shape[0], -1)
eight_by_eight_image = digits.images[0]


import math 
def sigmoid(z):
    
    '''
    The sigmoid activation function for a single node of a neural net
    parameters:
        z: the result of a linear equation consisting of a set of feature inputs
        multiplied by a set of weights plus a bias.
    returns:
        a number between 0 and 1
    
    '''
    
    return 1/(1+math.e**(-z))

def flatten_one_image():
    
    flat_image = np.array(digits.data[0]).reshape(-1, 1)
    
    return flat_image


def random_weights(rows=64, col=1):
    
    '''
    Create a random array of weights for one perceptron. 
    
    rows: the number of weights, aligned with the number of inputs
    cols: corresponds to the number of nodes per layer
    
    returns: random initial weights for nodes in a given layer
    '''
    
    weights = np.random.uniform(-1,1,rows)
    return weights.reshape(-1,1)
    
    

def transfer(flattened_image, weights):
    
    '''
    Parameters:
    flattened_image: Pixels of an image flattened into one column
    weights: initial weights associated with each pixel
    
    returns: the dot product of the pixel array and the weights, 
    aka the result of the collector function.
    '''
    
    return weights.T.dot(flat_image)
