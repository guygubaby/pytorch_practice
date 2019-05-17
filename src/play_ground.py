import numpy
import torch



if __name__ == '__main__':
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

    a=torch.randn(3,4)
    print(a)
    print(torch.max(a,1)[1])
    print(torch.argmax(a,1).data.squeeze())

    # print(a.size(0))