from functions.PyTorchGN import NN_GN
from neural_networks.Conv2 import Conv2
from neural_networks.Conv2_classifier import Conv2_Classifier
from neural_networks.Conv3_classifier import Conv3_Classifier
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
    N=1400
    (train_X, train_y), (test_X, test_y) = get_data_classifier(N=N, linear_input=False)

    ## Get initial neural network parameters 
    m = train_y.shape[1]
    n = train_X.shape[1]

    neural_network = Conv3_Classifier()
    #summary(neural_network, input_size = (1, 28, 28))
    neural_network.float()
    nn_gn = NN_GN(neural_network)
    X0 = nn_gn.get_X()
    print("Parameters", X0.shape)

    ## Do Gauss Newton
    MAX_TIME = 60
    EXPERIMENTS_COUNT = 1

    losses = []

    labels = []
    plt.figure()
    for _ in range(EXPERIMENTS_COUNT):
        X_est,train_errors, _ , timer1, _,= optimize(nn_gn, X0, train_X, train_y, batch_size=300, max_time=MAX_TIME, backtrack=True,
                        optimization_method="Random columns", optim_params={"keep_prob":0.1, "momentum": 0}, visualize=False)
        Ts = np.linspace(0, MAX_TIME, len(train_errors))
        plt.plot(Ts, train_errors, color='b')

        # X_est,train_errors, _ , timer1, _,= optimize(nn_gn, X0, train_X, train_y, batch_size=300, max_time=MAX_TIME, backtrack=True,
        #                 optimization_method="Random columns", optim_params={"keep_prob":0.01, "momentum": 0}, visualize=False)

        # Ts = np.linspace(0, MAX_TIME, len(train_errors))
        # plt.plot(Ts, train_errors, color='g')
        sgd_losses_lr05 = stochastic_gradient_descent(train_X, train_y,  nn_gn=nn_gn, max_time=MAX_TIME, batch_size=300, lr=0.5)

        Ts = np.linspace(0, MAX_TIME, len(sgd_losses_lr05))
        plt.plot(Ts, sgd_losses_lr05, color='r')

    




    plt.show()