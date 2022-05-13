import numpy as np

def get_accuracy(g, A, Y, X_est):
    correct = 0
    for a, y_true in zip(A, Y):
        y_est = g.forward(a, X_est)
        if y_est.shape[0] >1:
            y_est = np.argmax(y_est)
            y_true = np.argmax(y_true)
            if y_est == y_true:
                correct+=1.
        else: 
            if np.linalg.norm(y_true-y_est, 2) < 0.5:
                correct+=1.
    return correct/A.shape[0]