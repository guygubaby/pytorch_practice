import torch
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torch.autograd import Variable


def load_data():
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    train_loader=Data.DataLoader(dataset=train_data,batch_size=50,shuffle=True)

    test_data=torchvision.datasets.MNIST(
        root='./mnist',
        train=False
    )
    test_x=(torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)/255)[:2000]
    test_y=test_data.targets[:2000]
    return train_loader,test_x,test_y


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input shape (1, 28, 28)
        self.conv1=torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=5,
                            stride=1,
                            padding=2),
            ## if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # out 16*14*14
        self.conv2=torch.nn.Sequential(
            torch.nn.Conv2d(16,32,5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        # out 32*7*7
        self.out=torch.nn.Linear(32*7*7,10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1) # flattern
        out=self.out(x)
        return out


if __name__ == '__main__':
    train_loader, test_x,test_y=load_data()
    print(test_x.size(),test_y.size())

    cnn=CNN()
    print(cnn)

    optimizer=torch.optim.RMSprop(cnn.parameters(),alpha=0.9)
    loss_func=torch.nn.CrossEntropyLoss()

    for epoch in range(1):
        for step,(x,y) in enumerate(train_loader):
            batch_x=Variable(x)
            batch_y=Variable(y)

            out=cnn(batch_x)

            loss=loss_func(out,batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step%200==0:
                test_output=cnn(test_x)
                # print(test_output.size())
                predict_y=torch.argmax(test_output,1).data.squeeze()
                acc=(predict_y==test_y).sum().item()/float(test_y.size(0))

                print(f'epoch : {epoch} , loss : {loss} , acc : {acc}')

    test_out=cnn(test_x[:10])
    # print(test_out.size())
    pred_out=torch.argmax(test_out,1).data.squeeze()
    print('predict number ->',pred_out.numpy())
    print('real number ->',test_y[:10].numpy())
