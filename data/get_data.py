
from keras.datasets import mnist
import numpy as np


def get_data(N):
    num = 8
    ## Load MNIST Dataset 
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    ## Normalize data
    train_X = train_X/np.max(train_X)
    test_X = test_X/np.max(train_X)


    train_y = np.array(train_y,dtype=np.float32)
    train_X = np.array(train_X, dtype=np.float32).reshape((-1, 28*28))
    _, n = train_X.shape

    neg_idx = np.where(train_y==num)
    pos_idx = np.where(train_y!=num)

    train_y_pos = train_y[pos_idx]
    train_X_pos = train_X[pos_idx]
    train_y_neg = train_y[neg_idx]
    train_X_neg = train_X[neg_idx]

    ## Pick first N datapoints
    train_X = np.vstack((
        train_X_pos[:N//2], 
        train_X_neg[:N//2]
    ))
    train_y = np.hstack((
        train_y_pos[:N//2], 
        train_y_neg[:N//2]
    )).reshape((N, 1))

    test_X = np.vstack((
        train_X_pos[N//2:N], 
        train_X_neg[N//2:N]
    ))
    test_y = np.hstack((
        train_y_pos[N//2:N], 
        train_y_neg[N//2:N]
    )).reshape((N, 1))
    
    assert(train_X.shape==(N, n))
    assert(train_y.shape==(N, 1)), train_y.shape
    assert(test_X.shape==(N, n))
    assert(test_y.shape==(N,1))
    ## Convert target labels to binary classification
    train_y_new = np.zeros_like(train_y)
    test_y_new = np.zeros_like(test_y)

    for i, (y, y_test) in enumerate(zip(train_y, test_y)):
        if y == num:
            train_y_new[i] = np.array([1.])
        if y_test == num:
            test_y_new[i] = np.array([1.])
    train_y, test_y = train_y_new, test_y_new

    return (train_X, train_y), (test_X, test_y)