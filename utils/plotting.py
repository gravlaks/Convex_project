import matplotlib.pyplot as plt

def plot(train_errors, test_errors):
    plt.plot(train_errors, label = "Train loss")
    plt.plot(test_errors, label="Test loss")
    plt.legend()
    plt.show()