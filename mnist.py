from keras.datasets import mnist
import sys
sys.path.append("./functions")
from modular_nn import NN, get_initial_params
# from functions.modular_nn import NN, get_initial_params
import numpy as np
import matplotlib as mpl
mpl.use('MacOSX')
from matplotlib import pyplot as plt
from algorithms.algo2 import optimize
from data.get_data import get_data
import torch


if __name__ == '__main__':
    torch.manual_seed(0)

    ## Load MNIST Dataset 
    N=1000
    (train_X, train_y), (test_X, test_y) = get_data(N=N)
    print(test_y.shape)
    ## Get initial neural network parameters 
    m = train_y.shape[1]
    n = train_X.shape[1]
    #X0 = get_initial_params(hidden_layer_count=2, m=m, n=n, hidden_neurons=2)
    X0 = get_initial_params(hidden_layer_count=2, m=m, n=n, hidden_neurons=50)

    #Construct neural network with initial parameter setup
    nn = NN(X0)
    print(nn.param_count)
    ## Do Gauss Newton
    X_est,train_errors, test_errors = optimize(nn, nn.flatten(X0), train_X, train_y, X_test = test_X, Y_test = test_y)
    plt.plot(train_errors, label = "Train errors")
    plt.plot(test_errors, label="Test errors")
    plt.legend()
    # plt.show()
    
    ## Print results
    correct = 0
    for a, y_true in zip(train_X, train_y):
        y_est = nn.forward(a, X_est)
        if np.linalg.norm(y_true-y_est, 2) < 0.5:
            correct+=1.
    print("Train Accuracy", correct/N)

    correct = 0
    for a, y_true in zip(test_X, test_y):
        y_est = nn.forward(a, X_est)
        if np.linalg.norm(y_true-y_est, 2) < 0.5:
            correct+=1.
    print("Test Accuracy", correct/N)