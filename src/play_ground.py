import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


if __name__ == '__main__':
    a=torch.linspace(-10,10,10)
    print(a)
    print(torch.log(a))

    # x=torch.linspace(-10,10,100)
    # y=torch.tanh(x)
    # y1=torch.sigmoid(x)
    # plt.figure()
    # plt.plot(x.numpy(),y.numpy())
    # plt.plot(x.numpy(),y1.numpy())
    # plt.legend(['tanh','sigmoid'])
    # plt.show()

    # a=np.linspace(-10,10,10)
    # b=np.linspace(-100,100,10)
    # print(a.shape,b.shape,zip(a,b))
    # for i,j in zip(a,b):
    #     print(i,j)

    # a=torch.linspace(-1,1,10)
    # print(a,a.size())
    # b=torch.unsqueeze(a,dim=1)
    # print(b,b.size())
    # c=torch.squeeze(b,dim=1)
    # print(c,c.size())

    # a=torch.arange(1,10)
    # print(a)
    # b=torch.arange(1,0,-0.1)
    # print(b)


    # a=torch.randn(50,1,28,28)
    # print(a.size())

    # a=torch.randn(3,4)
    # print(a)
    # print(torch.max(a,1)[1])
    # print(torch.argmax(a,1).data.squeeze())

    # print(a.size(0))
