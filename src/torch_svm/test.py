import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input=torch.nn.Linear(2,10)
        self.out=torch.nn.Linear(10,2)

    def forward(self, x):
        x=F.relu(self.input(x))
        x=self.out(x)
        return x


if __name__ == '__main__':
    init_data=torch.ones(100,2)
    # print(init_data)
    x0=torch.normal(init_data*2,1)
    # print(x0)
    y0=torch.zeros(100)

    x1=torch.normal(-1*init_data,1)
    y1=torch.ones(100)

    x=torch.cat((x0,x1),0).type(torch.FloatTensor)
    y=torch.cat((y0,y1)).type(torch.LongTensor)

    print(x.size(),y.size())

    plt.scatter(x.data.numpy()[:,0],x.data.numpy()[:,1],c=y.data.numpy(),s=50,lw=0, cmap='RdYlGn')
    plt.show()

    net=Net()
    optimizer=torch.optim.RMSprop(net.parameters(),lr=0.02,alpha=0.9)

    loss_func=torch.nn.CrossEntropyLoss()

    for i in range(1000):
        out=net(x)
        # print(out,y)
        loss=loss_func(out,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10==0:
            print(f'loss is : {loss}')