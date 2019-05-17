import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def load_data(fname):
    data=[]
    with open(fname) as f:
        label_line=f.readline()
        for line in f.readlines():
            data.append(list(map(float,line.strip().split())))
    data=np.asarray(data)
    # print(data)
    x=data[:,:2]
    y=data[:,2]
    # print(np.mean(x,axis=0))
    x=x/np.mean(x,axis=0)
    x=torch.from_numpy(x).type(torch.FloatTensor)
    y=torch.from_numpy(y).type(torch.LongTensor)
    # print(x)
    # final_data=torch.Tensor(data)
    # x=final_data[:,:2].type(torch.FloatTensor)
    # y=final_data[:,2].type(torch.LongTensor)
    return x,y


class SVMNet(torch.nn.Module):
    def __init__(self,n_input,n_hidden,n_out):
        super(SVMNet, self).__init__()
        self.hidden=torch.nn.Linear(n_input,n_hidden)
        self.out=torch.nn.Linear(n_hidden,n_out)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.out(x)
        return x


if __name__ == '__main__':
    train_file='data/train_linear.txt'
    test_file='data/test_linear.txt'

    # load_data(train_file)

    x_train,y_train=load_data(train_file)

    plt.scatter(x_train.data.numpy()[:,0],x_train.data.numpy()[:,1],c=y_train.data.numpy(),s=50,lw=0,cmap='RdYlGn')
    plt.show()

    svm_net=SVMNet(2,10,2)

    optimizer=torch.optim.SGD(svm_net.parameters(),lr=0.02)
    loss_func=torch.nn.CrossEntropyLoss()

    for i in range(1):
        out=svm_net(x_train)
        print(out,y_train)

        loss=loss_func(out,y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%10==0:
            print('loss : {}'.format(loss))

    x_test,y_test=load_data(test_file)
    prediction=svm_net(x_test)
    print(prediction)




