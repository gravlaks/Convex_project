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
    MAX_TIME = 20
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    X_est,train_errors_1, _ = optimize(nn_gn, X0, train_X, train_y, steps=1000, batch_size=200, max_time=MAX_TIME, backtrack=True,
                        optimization_method="Random columns", optim_params={"keep_prob":1})

    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    X_est,train_errors_05, _ = optimize(nn_gn, X0, train_X, train_y, steps=1000, batch_size=200, max_time=MAX_TIME, backtrack=True,
                        optimization_method="Random columns", optim_params={"keep_prob":0.5})
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    X_est,train_errors_01, _ = optimize(nn_gn, X0, train_X, train_y, steps=1000, batch_size=200, max_time=MAX_TIME, backtrack=True,
                        optimization_method="Random columns", optim_params={"keep_prob":0.1})
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    X_est,train_errors_001, _ = optimize(nn_gn, X0, train_X, train_y, steps=1000, batch_size=200, max_time=MAX_TIME, backtrack=True,
                        optimization_method="Random columns", optim_params={"keep_prob":0.01})
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))

    sgd_losses_lr05 = stochastic_gradient_descent(train_X, train_y, epochs=3000, nn_gn=nn_gn, max_time=MAX_TIME, batch_size=200, lr=0.5)
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    
    sgd_losses_lr01 = stochastic_gradient_descent(train_X, train_y, epochs=3000, nn_gn=nn_gn, max_time=MAX_TIME, batch_size=200, lr=0.1)
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    sgd_losses_lr001 = stochastic_gradient_descent(train_X, train_y, epochs=3000, nn_gn=nn_gn, max_time=MAX_TIME, batch_size=200, lr=0.01)
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    sgd_losses_lr0001 = stochastic_gradient_descent(train_X, train_y, epochs=3000, nn_gn=nn_gn, max_time=MAX_TIME, batch_size=200, lr=0.001)

    ## Print results
    losses = [train_errors_1,train_errors_05, train_errors_01, train_errors_001, sgd_losses_lr05, sgd_losses_lr01, sgd_losses_lr001, sgd_losses_lr0001]
    labels = ["Algo 2: keep_prob 1", "Algo 2: keep_prob 0.5", "Algo 2: keep_prob 0.1", "Algo2 : keep_prob 0.01", "SGD lr: 0.5","SGD lr: 0.1", "SGD lr: 0.01", "SGD lr: 0.001"]
    print("Train Accuracy", get_accuracy(nn_gn, train_X, train_y))
    print("Test Accuracy", get_accuracy(nn_gn, test_X, test_y))
    plot_mult(losses, labels,  "plots/Classifier10dim", MAX_TIME, "Algo 2: Digit 10dim classification")

    



