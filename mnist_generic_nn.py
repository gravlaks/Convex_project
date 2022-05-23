from functions.PyTorchGN import NN_GN
from neural_networks.FC1 import FC1
from neural_networks.FC2 import FC2
from utils.evaluation import get_accuracy
from utils.plotting import plot
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algo1 import train_1
from data.get_data import get_data, get_data_classifier
import torch


if __name__ == '__main__':
    torch.manual_seed(0)

    ## Load MNIST Dataset 
    N=1000
    (train_X, train_y), (test_X, test_y) = get_data(N=N)

    ## Get initial neural network parameters 
    m = train_y.shape[1]
    n = train_X.shape[1]

    neural_network = FC2(input_dim=n, output_dim=m)
    nn_gn = NN_GN(neural_network)
    X0 = nn_gn.get_X()

    ## Do Gauss Newton

    #X_est,train_errors, test_errors = optimize(nn_gn, X0, train_X, train_y, X_test = test_X, Y_test = test_y, steps=100)
    X_est = train_1(nn_gn, train_X, train_y, x_init=X0, k=1)
    #plot(train_errors, test_errors)
    
    ## Print results
    
    #print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y, X_est))
    #print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y, X_est))

    



