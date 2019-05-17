import torch
from torch import nn
import torch.utils.data as Data
import torchvision.datasets as dsets
from torchvision.transforms import transforms
from torch.autograd import Variable


def load_data():
    # Mnist digital dataset
    train_data = dsets.MNIST(
        root='./mnist/',
        train=True,  # this is training data
        transform=transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
        # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
        download=True,  # download it if you don't have it
    )

    test_data=dsets.MNIST(
        root='./mnist/',
        train=False,
        transform=transforms.ToTensor()
    )

    test_x=Variable(test_data.data).type(torch.FloatTensor)[:2000]/255.
    test_y=test_data.targets.squeeze()[:2000]
    # print(test_data.targets.numpy().squeeze()[:2000].shape)

    train_loader=Data.DataLoader(dataset=train_data,shuffle=True,batch_size=64)

    return train_loader,test_x,test_y


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(
            input_size=28,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        # self.rnn = nn.LSTM(
        #     input_size=64,
        #     hidden_size=128,
        #     num_layers=1,
        #     batch_first=True
        # )

        self.out=nn.Linear(64,10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        r_out, (h_n, h_c)=self.rnn(x,None) # None represents zero initial hidden state
        # print('r_out -> ',r_out)
        # choose r_out at the last time step
        out = self.out(r_out[:,-1,:])
        return out


if __name__ == '__main__':
    train_loader, test_x, test_y=load_data()
    # print(test_x.size(),test_y.shape)

    rnn=RNN()
    print(rnn)

    optimizer=torch.optim.RMSprop(rnn.parameters(),alpha=0.9)
    loss_func=nn.CrossEntropyLoss()

    for epoch in range(1):
        for step,(x,y) in enumerate(train_loader):
            # print('x->',x.size())
            batch_x=Variable(x.view(-1,28,28)) # reshape x to (batch, time_step, input_size)
            # print('view x ->',batch_x.size())
            batch_y=Variable(y)

            out=rnn(batch_x)
            loss=loss_func(out,batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%100==0:
                test_out=rnn(test_x)
                predict_y=torch.argmax(test_out,dim=1).data.squeeze()

                acc = (predict_y==test_y).sum().item()/float(test_y.size(0))

                print(f'loss is {loss} , acc is : {acc}')

    test_output=rnn(test_x[:10])
    predict_test=torch.argmax(test_output,dim=1).data.squeeze()
    print(f'prediction : {predict_test}')
    print(f'true value : {test_y[:10]}')