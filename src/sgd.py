from datetime import date
from time import time
import torch.optim as optim
import torch.nn as nn
from functions.PyTorchGN import NN_GN
from neural_networks.FC2 import FC2
from data.get_data import get_data, get_data_classifier
import torch
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def stochastic_gradient_descent(X, Y,  nn_gn, max_time, batch_size, lr=0.001):
    net = nn_gn.nn
    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

   

    net.apply(weight_reset)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    criterion = nn.MSELoss()
    losses = []
    t1 = datetime.now()
    epoch = 1
    while datetime.now()-t1<=timedelta(seconds=max_time):  

        
        random_indices = np.random.choice(X.shape[0],
                                  size=batch_size,
                                  replace=False)
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = X[random_indices], Y[random_indices]
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}')
        inputs_l, labels_l = torch.tensor(X), torch.tensor(Y)
        out_l = net(inputs_l)
        mse_loss = criterion(out_l, labels_l)

        l = customloss(nn_gn, X, Y)
        if epoch%100==1:
            
            print(f"{epoch}: Loss MSE: {l} ")
        epoch+=1
        losses.append(mse_loss.detach())
    print("Timeout")
    return losses


def customloss(g, A, Y):
    X = g.get_X()

    return np.sum(
        #[np.linalg.norm(g.forward(a, X)-y)**2 for a, y in zip(A, Y)]
        np.linalg.norm(g.forward(A, X).reshape((-1, 1))-Y.reshape((-1, 1)), axis=1)**2
    )/A.shape[0]

if __name__=='__main__':


    
    N=2500
    (train_X, train_y), (test_X, test_y) = get_data(N=N)

    ## Get initial neural network parameters 
    m = train_y.shape[1]
    n = train_X.shape[1]
    net = FC2(input_dim=n, output_dim=m)    
    nn_gn = NN_GN(net)

    losses = stochastic_gradient_descent(train_X, train_y, epochs=300, nn_gn=nn_gn, max_time=10, batch_size=100)
    plt.plot(losses)