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
    
    plt.figure()


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

    #plt.show()

def plot_timer(timer, title="Timer"):
    plt.figure()
    y = [np.mean(np.array(timer["Jac creation"])), np.mean(np.array(timer["LS solve"])), np.mean(np.array(timer["backtrack"]))]
    
    plt.pie(y, labels = ["Jacobian creation", "LS solve", "Backtracking"])
    plt.title(title)
    plt.savefig("plots/"+title+".png")

def visualize_step(step):
    plt.hist(step, bins=100,density=True, log=True)
    plt.show()

def plot_backtracks(bactracks):
    plt.figure()
    plt.plot(bactracks)
    plt.title("Backtracks per iteration")
    plt.show()