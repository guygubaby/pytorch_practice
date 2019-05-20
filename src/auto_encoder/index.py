import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def load_data():
    train_data=torchvision.datasets.MNIST(
        root='./mnist/',
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # print(train_data.data.size(),train_data.targets.size())
    # print(type(train_data.targets[2].data.numpy()))

    # plt.imshow(train_data.data[2].numpy(),cmap='gray')
    # plt.title(train_data.targets[2].numpy())
    # plt.show()

    train_loader=Data.DataLoader(dataset=train_data,batch_size=100,shuffle=True)
    return train_loader,train_data


class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder=torch.nn.Sequential(
            nn.Linear(28*28,128),
            nn.Tanh(),
            nn.Linear(128,64),
            nn.Tanh(),
            nn.Linear(64,12),
            nn.Tanh(),
            nn.Linear(12,3)
            # compress to 3 features which can be visualized in plt
        )
        self.decoder=nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoder=self.encoder(x)
        decoder=self.decoder(encoder)
        return encoder,decoder


if __name__ == '__main__':
    train_loader,train_data=load_data()
    net=AutoEncoder()
    print(net)

    view_data=Variable(train_data.data[:5].view(-1,28*28).type(torch.FloatTensor)/255.)
    print('view data : ',view_data.size())
    optimizer=torch.optim.RMSprop(net.parameters(),alpha=0.9)
    loss_func=torch.nn.MSELoss()

    for epoch in range(10):
        for step,(x,y) in enumerate(train_loader):
            batch_x=Variable(x.view(-1,28*28))
            batch_y=Variable(x.view(-1,28*28))

            label_y=Variable(y)
            encoder,decoder=net(batch_x)

            loss=loss_func(decoder,batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%1000==0:
                print(f'train loss is : {loss}')
                _,out=net(view_data)

                f,a=plt.subplots(2,5,figsize=(5,2))

                for i in range(5):
                    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

                for i in range(5):
                    a[1][i].clear()
                    a[1][i].imshow(np.reshape(out.data.numpy()[i], (28, 28)), cmap='gray')
                    a[1][i].set_xticks(())
                    a[1][i].set_yticks(())

                plt.show()
                plt.pause(0.05)

    squere_data=Variable(train_data.data[:200].view(-1,28*28).type(torch.FloatTensor)/255.)
    encoded_data,_=net(squere_data)

    # fig=plt.figure(2)
    # ax=Axes3D(fig)
    # X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
    # values = train_data.train_labels[:200].numpy()
    # for x, y, z, s in zip(X, Y, Z, values):
    #     c = cm.rainbow(int(255 * s / 9))
    #     ax.text(x, y, z, s, backgroundcolor=c)
    #     ax.set_xlim(X.min(), X.max())
    #     ax.set_ylim(Y.min(), Y.max())
    #     ax.set_zlim(Z.min(), Z.max())
    # plt.show()