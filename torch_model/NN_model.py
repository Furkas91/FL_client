import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import pandas as pd
import numpy as np

from torch_model.DataLoader import get_data_loader


class UniversalNet(nn.Module):
    def __init__(self, layers, activations):
        super(UniversalNet, self).__init__()
        self.number_of_layers = len(layers)
        self.layers = nn.ModuleList(layers)
        self.activations = activations

    def forward(self, x):
        for i in range(self.number_of_layers):
            x = self.layers[i](x)
            x = self.activations[i](x)
        return x


def create_nn(batch_size=200, learning_rate=0.01, epochs=10,
              log_interval=10):
    # train_df = pd.read_csv('D:/FL_client/data/MNIST/train.csv')
    train_loader = get_data_loader('D:/FL_client/data/MNIST/train.csv', batch_size)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size, shuffle=True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28 * 28, 200)
            self.fc2 = nn.Linear(200, 200)
            self.fc3 = nn.Linear(200, 10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return F.log_softmax(x)

    layers = [
        nn.Linear(28*28, 200, bias=False),
        nn.Linear(200, 200),
        nn.Linear(200, 10)
    ]
    activations = [
        F.relu,
        F.relu,
        F.log_softmax
    ]
    net = UniversalNet(layers, activations)
    # net = Net()
    print(net)
    # create a stochastic gradient descent optimizer
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # create a loss function
    criterion = nn.NLLLoss()

    # run the main training loop
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28 * 28)

            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            # print(loss.data)
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    # run a test loop
    test_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 28 * 28)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data.item()
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(train_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    return net


if __name__ == "__main__":
    net  = create_nn()
