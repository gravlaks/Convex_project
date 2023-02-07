from functions.PyTorchGN import NN_GN
from neural_networks.BacktrackFC import BacktrackFC
from neural_networks.FC1 import FC1
from neural_networks.FC2 import FC2
from neural_networks.Conv1 import Conv1
from sgd import stochastic_gradient_descent
from utils.evaluation import get_accuracy
from utils.plotting import plot, plot_mult
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algo1 import train_1
from algorithms.algo2 import optimize
from data.get_data import get_data, get_data_classifier
import torch


if __name__ == '__main__':
    torch.manual_seed(0)

    ## Load MNIST Dataset 
    N=1000
    MAX_TIME=20



    (train_X, train_y), (test_X, test_y) = get_data(N=N)

    ## Get initial neural network parameters 
    m = train_y.shape[1]
    n = train_X.shape[1]
    print("Dimensions", m, n)

    neural_network = BacktrackFC(input_dim=n, output_dim=m)
    nn_gn = NN_GN(neural_network)
    X0 = nn_gn.get_X()



    ## Do Gauss Newton
    print("Initial train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Initial test Accuracy", get_accuracy(nn_gn, test_X, test_y))

    X_est,losses_gn_backtrack, _, _ = optimize(nn_gn, X0, train_X, train_y, max_time=MAX_TIME, batch_size=300 ,backtrack=True)
    print("Train Accuracy GN Proj.", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy GN Proj.", get_accuracy(nn_gn, test_X, test_y))


    X_est,losses_gn, _, _= optimize(nn_gn, X0, train_X, train_y, max_time=MAX_TIME, batch_size=300, backtrack=False)
    #X_est = train_1(nn_gn, train_X, train_y, x_init=X0, k=1)
    print("Train Accuracy GN", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy GN", get_accuracy(nn_gn, test_X, test_y))
    losses = [losses_gn_backtrack, losses_gn]
    labels = ["With Backtrack", "Without Backtrack"]
    print(nn_gn.param_count)
    plot_mult(losses, labels, "plots/Backtrackcomparison", MAX_TIME, "Algo 2 Backtracking Comparison")
        


    



