from keras.datasets import mnist
from modular_nn import NN, get_initial_params
import numpy as np
from optimize import optimize
from tensorflow.keras.utils import to_categorical
import torch
if __name__ == '__main__':
    torch.manual_seed(0)
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_y = np.array(train_y,dtype=np.float32)
    train_X = np.array(train_X, dtype=np.float32)

    N = 100
    train_X, train_y = train_X[:N].reshape((N, -1)), train_y[:N].reshape((-1, 1))
    train_y_new = np.zeros_like(train_y)
    for i, y in enumerate(train_y):
        if y == 0:
            train_y_new[i] = np.array([1.])

    #train_y =  to_categorical(train_y)
    #print(train_y)
    train_y = train_y_new
    m = train_y.shape[1]
    n = train_X.shape[1]
    X0 = get_initial_params(hidden_layer_count=2, m=m, n=n, hidden_neurons=2)
    nn = NN(X0)

    X_est = optimize(nn, nn.flatten(X0), train_X, train_y)
    print(nn.unflatten(X_est))
    correct = 0
    for a, y_true in zip(train_X, train_y):
        y_est = nn.forward(a, X_est)
        print("Y_true", y_true, "Y_est", y_est)
        if np.linalg.norm(y_true-y_est, 2) < 1e-2:
            correct+=1.
    print("Accuracy", correct/N)