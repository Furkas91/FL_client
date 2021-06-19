"""
Файл содержит код для механизма работы с моделью нейронной сети PyTorch
"""
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_model.DataLoader import get_data_loader

loss_functions = {
    ''
}

class MiningSettings:
    """
    Класс для сериализации настроек алгоритма для
    обучения нейронной сети
    """
    def __init__(self, algorithm, loss_function, epochs, learning_rate, momentum, batch_size):
        self.algorithm = algorithm
        self.loss_function = loss_function
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

class UniversalNet(nn.Module):
    """
    Класс универсальной модели
    """
    def __init__(self, layers, activations, normalizations):
        super(UniversalNet, self).__init__()
        self.number_of_layers = len(layers)
        self.layers = nn.ModuleList(layers)
        self.activations = activations
        self.normalizations = normalizations

    def forward(self, x):
        """
        Метод для прямого прохода по нейронной сети
        :param x: данные для прохода
        :return: данные после прохода
        """
        for i in range(self.number_of_layers):
            # Прохождения слоя
            x = self.layers[i](x)
            # Применение функции активации
            x = self.activations[i](x)
            # Применение нормализации
            x = self.normalizations[i](x)
        return x


def evaluate(test_loader, criterion, net):
    """
    Функция оценки точности модели
    :param test_loader: DataLoader
    :param criterion: функция потерь
    :param net: UniversalNet
    :return: точность и ошибка
    """
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data, volatile=True), Variable(target)
        data = data.view(-1, 40 * 6)
        net_out = net(data)
        # sum up batch loss
        test_loss += criterion(net_out, target).data.item()
        pred = net_out.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f})\n'.format(
        test_loss, correct, len(test_loader.dataset),
        correct / len(test_loader.dataset)))


def fit(net, train_loader, epochs, criterion, optimizer):
    """
    Функция, тренировки модели
    :param net: UniversalNet
    :param train_loader: DataLoader
    :param epochs: количество эпох
    :param criterion: функция потерь
    :param optimizer: метод оптимизации
    :return: UniversalNet
    """
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            data = data.view(-1, 40*6)

            optimizer.zero_grad()
            net_out = net(data)
            loss = criterion(net_out, target)
            loss.backward()
            optimizer.step()
            # print(loss.data)
            #if batch_idx % 100 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))
    return net


def train_evaluate(net, path='D:/FL_client/data/MNIST/train.csv', settings='nothing', data_name='MNIST'):
    """
    Функция для обучения и оценки точности модели
    :param net: UniversalNet
    :param path: путь к CSV файлу
    :param settings: MiningSettings
    :param data_name: название набора данных
    :return: UniversalNet
    """
    # Загрузка набора данных
    train_loader = get_data_loader(path, settings.batch_size, data_name=data_name)
    # Создание объекта для метода оптимизации
    optimizer = optim.SGD(net.parameters(), lr=settings.learning_rate, momentum=settings.momentum)
    # Создание объекта для функции потерь
    criterion = settings.loss_function()
    # Вызов функции обучения
    net = fit(net, train_loader, settings.epochs, criterion, optimizer)
    # Вызов функции оценки точности модели
    evaluate(train_loader, criterion, net)
    return net, len(train_loader)


if __name__ == "__main__":
    layers = [
        nn.Linear(240, 200),
        nn.Linear(200, 200),
        nn.Linear(200, 3)
    ]
    activations = [
        F.relu,
        F.relu,
        F.log_softmax
    ]
    normalizations = [
        nn.BatchNorm1d(200),
        nn.BatchNorm1d(200),
        lambda x: x
    ]
    settings = MiningSettings(
        algorithm='SGD',
        loss_function=nn.NLLLoss,
        epochs=10,
        learning_rate=0.1,
        momentum=0.9,
        batch_size=10
    )
    net = UniversalNet(layers, activations, normalizations)
    net = train_evaluate(net=net, path='D:\FL_client\data\smartilizer\Video-11-15-40-560.csv', data_name='smartiliser', settings=settings)
