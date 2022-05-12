from functions.PyTorchGN import NN_GN
from testing_jacobian.jacobian_pytorch import Net

import numpy as np
import matplotlib.pyplot as plt
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
    #Construct neural network with initial parameter setup
    neural_network = Net(input_dim=n, output_dim=m)
    nn_gn = NN_GN(neural_network)
    X0 = nn_gn.get_X()
    ## Do Gauss Newton

    X_est,train_errors, test_errors = optimize(nn_gn, X0, train_X, train_y, X_test = test_X, Y_test = test_y)
    plt.plot(train_errors, label = "Train errors")
    plt.plot(test_errors, label="Test errors")
    plt.legend()
    plt.show()
    
    ## Print results
    correct = 0
    for a, y_true in zip(train_X, train_y):
        y_est = nn_gn.forward(a, X_est)
        if np.linalg.norm(y_true-y_est, 2) < 0.5:
            correct+=1.
    print("Train Accuracy", correct/N)

    correct = 0
    for a, y_true in zip(test_X, test_y):
        y_est = nn_gn.forward(a, X_est)
        if np.linalg.norm(y_true-y_est, 2) < 0.5:
            correct+=1.
    print("Test Accuracy", correct/N)



