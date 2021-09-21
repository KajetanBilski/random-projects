import sys
import os
import pts_loader
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pickle
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyAutoEncoder(nn.Module):

    def __init__(self, cloud_size):
        super().__init__()
        self.cloud_size = cloud_size
        self.encoder = torch.nn.Sequential([
            nn.Conv1d(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128),
            nn.BatchNorm1d(128)
        ])
        self.decoder = nn.Sequential([
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.cloud_size * 3)
        ])
    
    def encode(self, x):
        return torch.max(self.encoder(x), 2)[0]
    
    def decode(self, x):
        return self.decoder(x).view(-1, self.cloud_size, 3)

    def forward(self, x):
        return self.decode(self.encode(x))

def load_data(type):
    path1 = './' + type + '_data/'
    clouds = []
    for dir in os.listdir(path1):
        path2 = path1 + dir + '/'
        for file in os.listdir(path2):
            path3 = path2 + file
            clouds.append(pts_loader.load(path3))
    return clouds

def save_net(filename, state_dict):
    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(state_dict, f)

def train(train_dl: DataLoader, val_dl: DataLoader, model: nn.Module, optimizer: torch.optim.Optimizer, loss, epochs = 100, save_file = ''):
    try:
        for epoch in range(epochs):
            for batch in tqdm(train_dl):
                batch = batch.to(device)

    except KeyboardInterrupt:
        if save_file:
            save_net(save_file, model.state_dict())
        raise KeyboardInterrupt

def main(argv):
    pass

if __name__ == '__main__':
    main(sys.argv)
