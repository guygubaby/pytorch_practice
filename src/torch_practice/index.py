import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.output=torch.nn.Linear(n_hidden,n_output)

    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=F.relu(self.output(x))
        return x


if __name__ == '__main__':
    torch.manual_seed(1)
    x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
    y=x.pow(2)+0.1*torch.randn(x.size())

    plt.figure()
    plt.scatter(x,y)
    plt.show()

    net=Net(1,10,1)
    print(net)

    optmizer=torch.optim.SGD(net.parameters(),lr=0.2)
    loss_func=torch.nn.MSELoss()

    for i in range(1000):
        prediction=net(x)
        loss=loss_func(prediction,y)
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()

        if i%100==0:
            print(f'loss {loss}')