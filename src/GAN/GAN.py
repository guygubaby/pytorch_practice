import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn


def artist_works():     # painting from the famous artist (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return Variable(paintings)


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.G=nn.Sequential(
            nn.Linear(N_IDEAS,128),
            nn.ReLU(),
            nn.Linear(128,ART_COMPONENTS)
        )
        self.D=nn.Sequential(
            nn.Linear(ART_COMPONENTS,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g_out=self.G(x)
        d_out=self.D(g_out)
        return d_out


if __name__ == '__main__':
    # Hyper Parameters
    BATCH_SIZE = 64
    LR_G = 0.0001  # learning rate for generator
    LR_D = 0.0001  # learning rate for discriminator
    N_IDEAS = 5  # think of this as number of ideas for generating an art work (Generator)
    ART_COMPONENTS = 15  # it could be total point G can draw in the canvas
    PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

    # show our beautiful painting range
    # plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
    # plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
    # plt.legend(loc='best')
    # plt.show()

    res=artist_works()
    # print(res.size())

    net=GAN()
    print(net)

    opt_G=torch.optim.RMSprop(net.G.parameters(),alpha=0.9)
    opt_D=torch.optim.RMSprop(net.D.parameters(),alpha=0.9)

    for step in range(10000):
        artist_paintings = artist_works()  # real painting from artist
        G_ideas = Variable(torch.randn(BATCH_SIZE, N_IDEAS))  # random ideas
        G_paintings = net.G(G_ideas)  # fake painting from G (random ideas)

        prob_artist0 = net.D(artist_paintings)  # D try to increase this prob
        prob_artist1 = net.D(G_paintings)  # D try to reduce this prob

        D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
        G_loss = torch.mean(torch.log(1. - prob_artist1))

        opt_D.zero_grad()
        D_loss.backward(retain_graph=True)  # retain_variables for reusing computational graph
        opt_D.step()

        opt_G.zero_grad()
        G_loss.backward()
        opt_G.step()

        if step % 1000 == 0:  # plotting
            plt.cla()
            plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
            plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
            plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
            plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                     fontdict={'size': 15})
            plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 15})
            plt.ylim((0, 3))
            plt.legend(loc='upper right', fontsize=12)
            plt.draw()
            plt.pause(0.01)
            plt.show()
