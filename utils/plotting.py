import matplotlib.pyplot as plt
import numpy as np

def plot(train_errors, sgd_loss):
    plt.plot(train_errors, label = "Train loss")
    idx = np.round(np.linspace(0, len(sgd_loss) - 1, len(train_errors))).astype(int)
    plt.plot(np.array(sgd_loss)[idx], label="SGD Losses")
    plt.semilogy()
    plt.legend()
    plt.show()