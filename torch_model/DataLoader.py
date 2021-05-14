import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


def best_transform(data):
    return data


def mnist_transform():
    transform = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.1307,), (0.3081,))
    ])
    return transform


class TrainDataset(Dataset):
    def __init__(self, features, labels, Transform):
        self.x = features
        self.y = labels
        self.transform = Transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.transform(self.x[index]), self.y[index]


def GetDf(df, Transform=best_transform):
    x_features = df.iloc[:, 1:].values
    y_labels = df.label.values
    x_features = x_features.reshape(-1, 1, 28, 28)
    x_features = np.uint8(x_features)
    x_features = torch.from_numpy(x_features)
    y_labels = torch.from_numpy(y_labels)
    return TrainDataset(x_features, y_labels, Transform)


def get_data_loader(path, batch_size, transform=mnist_transform()):
    train_df = pd.read_csv(path)

    train_loader = torch.utils.data.DataLoader(
        GetDf(train_df,  transform),
        batch_size=batch_size, shuffle=True)

    return train_loader
