import matplotlib.pyplot as plt
import numpy as np

def plot(train_errors, sgd_loss):
    
    plt.plot(train_errors, label = "Train loss")
    idx = np.round(np.linspace(0, len(sgd_loss) - 1, len(train_errors))).astype(int)
    plt.plot(np.array(sgd_loss)[idx], label="SGD Losses")
    plt.semilogy()
    plt.legend()
    plt.show()

def plot_mult(losses, labels, filename, tf, title):
    



    for loss, label in zip(losses, labels):
        Ts = np.linspace(0, tf, len(loss))
        #idx = np.round(np.linspace(0, len(loss) - 1, len(losses[0]))).astype(int)
        plt.plot(Ts, np.array(loss), label=label)
        plt.semilogy()
        plt.legend()
    plt.xlabel("Seconds")
    plt.ylabel("MSE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)

    plt.show()