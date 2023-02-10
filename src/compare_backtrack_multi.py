from functions.PyTorchGN import NN_GN
from neural_networks.Conv2 import Conv2
from neural_networks.Conv2_classifier import Conv2_Classifier
from neural_networks.FC1 import FC1
from neural_networks.FC2 import FC2
from neural_networks.Conv1 import Conv1
from sgd import stochastic_gradient_descent
from utils.evaluation import get_accuracy
from utils.plotting import *
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algo2 import optimize
from data.get_data import get_data, get_data_classifier
import torch


if __name__ == '__main__':
    torch.manual_seed(0)

    ## Load MNIST Dataset 
    N=1000
    (train_X, train_y), (test_X, test_y) = get_data_classifier(N=N, linear_input=False)
    ## Get initial neural network parameters 
    m = train_y.shape[1]
    n = train_X.shape[1]

    neural_network = Conv2_Classifier()
    neural_network.float()
    nn_gn = NN_GN(neural_network)
    X0 = nn_gn.get_X()
    print("Parameters", X0.shape)

    ## Do Gauss Newton
    MAX_TIME = 60
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    X_est,train_errors_gaussian, _, timer, backtracks = optimize(nn_gn, X0, train_X, train_y, batch_size=400, max_time=MAX_TIME, backtrack=False,
                        optimization_method="Gaussian", optim_params={"keep_prob":1})


    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    X_est,train_errors_05, _, timer, backtracks = optimize(nn_gn, X0, train_X, train_y, batch_size=400, max_time=MAX_TIME, backtrack=True,
                        optimization_method="Random columns", optim_params={"keep_prob":1})
    plot_backtracks(backtracks)
    ## Print results
    losses = [train_errors_gaussian,train_errors_05]
    labels = ["Algo 2: Without Backtrack", "Algo 2: With Backtrack"]
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    plot_mult(losses, labels,  "plots/backtrack", MAX_TIME, "Algo 2: Backtrack Comparison")
    



