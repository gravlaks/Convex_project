from functions.PyTorchGN import NN_GN
from neural_networks.Conv2 import Conv2
from neural_networks.Conv2_classifier import Conv2_Classifier
from neural_networks.FC1 import FC1
from neural_networks.FC2 import FC2
from neural_networks.Conv1 import Conv1
from sgd import stochastic_gradient_descent
from utils.evaluation import get_accuracy
from utils.plotting import plot, plot_mult
import numpy as np
import matplotlib.pyplot as plt
from algorithms.algo2 import optimize
from data.get_data import get_data, get_data_classifier
import torch
from torchsummary import summary


if __name__ == '__main__':
    torch.manual_seed(0)

    ## Load MNIST Dataset 
    N=1000
    (train_X, train_y), (test_X, test_y) = get_data_classifier(N=N, linear_input=False)

    ## Get initial neural network parameters 
    m = train_y.shape[1]
    n = train_X.shape[1]

    neural_network = Conv2_Classifier()
    #summary(neural_network, input_size = (1, 28, 28))
    neural_network.float()
    nn_gn = NN_GN(neural_network)
    X0 = nn_gn.get_X()
    print("Parameters", X0.shape)

    ## Do Gauss Newton
    MAX_TIME = 300

    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    X_est,train_errors, _ = optimize(nn_gn, X0, train_X, train_y, steps=1000, batch_size=200, max_time=MAX_TIME)
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))

    sgd_losses = stochastic_gradient_descent(train_X, train_y, epochs=300, nn_gn=nn_gn, max_time=MAX_TIME, batch_size=300)

    ## Print results
    losses = [train_errors, sgd_losses]
    labels = ["Algo 2", "SGD"]
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    plot_mult(losses, labels,  "plots/Classifier10dim", MAX_TIME, "Algo 2: Digit 10dim classification")

    



