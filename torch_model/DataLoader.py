import torch
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset
from torchvision import transforms


def best_transform(data):
    """
    Функция с ироничным названием и функционал,
    целью существования которой является ее
    применение к уже обработанным данным, но
    для структуры датасета необходима функция
    преобразования
    """
    return data


def mnist_transform():
    """
    Функция предназаченная для примения
    в датасете MNIST
    :return: функция преобразования
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    return transform


class TrainDataset(Dataset):
    """
    Класс, описывающий набор данных для обучения
    """
    def __init__(self, features, labels, Transform):
        self.x = features
        self.y = labels
        self.transform = Transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.transform(self.x[index]), self.y[index]


def get_mnist_data(df, Transform=best_transform):
    """
    Предобработка,для MNIST датасета
    """
    x_features = df.iloc[:, 1:].values
    y_labels = df.label.values
    x_features = x_features.reshape(-1, 1, 28, 28)
    x_features = np.uint8(x_features)
    x_features = torch.from_numpy(x_features)
    y_labels = torch.from_numpy(y_labels)
    return TrainDataset(x_features, y_labels, Transform)


def resize_data(X, y, time_steps=1, step=1):
    """
    Разбиение набора данных на пересекающие фреймы
    :param X:
    :param y:
    :param time_steps:
    :param step:
    :return:
    """
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        labels = y.iloc[i: i + time_steps]
        Xs.append(v)
        ys.append(stats.mode(labels)[0][0])
    return np.array(Xs), np.array(ys).reshape(-1)


def get_smartiliser_data(df, Transform=best_transform):
    """

    :param df:
    :param Transform:
    :return:
    """
    time_steps = 40
    step = 10
    df = df[::2]
    df = df[(df.activityMode == 5) | (df.activityMode == 6) | (df.activityMode == 7)]
    y_labels = df.activityMode - 5
    scale_columns = ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']
    df = df[scale_columns]
    scaler = RobustScaler()
    scaler = scaler.fit(df)
    df.loc[:, scale_columns] = scaler.transform(df[scale_columns].to_numpy())
    x_features = df
    x_features, y_labels = resize_data(x_features, y_labels, time_steps, step)
    x_features = x_features.reshape(len(x_features), -1)
    x_features = np.float32(x_features)
    return TrainDataset(x_features, y_labels, Transform)


def get_data_loader(path, batch_size, data_name='MNIST'):
    train_df = pd.read_csv(path)
    preprocces = {
        'MNIST': get_mnist_data,
        'smartiliser': get_smartiliser_data
    }
    transform = {
        'MNIST': mnist_transform(),
        'smartiliser': best_transform,
    }
    train_loader = torch.utils.data.DataLoader(
        preprocces[data_name](train_df, transform[data_name]),
        batch_size=batch_size, shuffle=True)

    return train_loader
