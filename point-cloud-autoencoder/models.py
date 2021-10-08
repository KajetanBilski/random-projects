import torch
from torch import nn
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyAutoEncoder(nn.Module):

    def __init__(self, cloud_size):
        super().__init__()
        self.cloud_size = cloud_size
        self.encoder = torch.nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128)
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.cloud_size * 3)
        )
    
    def encode(self, x):
        x = x.permute(0, 2, 1)
        return torch.max(self.encoder(x), 2)[0]
    
    def decode(self, x):
        return self.decoder(x).view(-1, self.cloud_size, 3)

    def forward(self, x):
        return self.decode(self.encode(x))


class MyVAutoEncoder(nn.Module):

    def __init__(self, cloud_size, latent_size=128):
        super().__init__()
        self.cloud_size = cloud_size
        self.latent_size = latent_size
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, self.latent_size, 1),
            nn.BatchNorm1d(self.latent_size)
        )
        self.encoder_mu = nn.Linear(self.latent_size, self.latent_size)
        self.encoder_var = nn.Linear(self.latent_size, self.latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.cloud_size * 3)
        )
    
    def encode(self, x):
        x = x.permute(0, 2, 1)
        x = torch.max(self.encoder(x), 2)[0]
        return self.encoder_mu(x), self.encoder_var(x)
    
    def decode(self, x):
        return self.decoder(x).view(-1, self.cloud_size, 3)
    
    def sample(self, mu, log_var):
        eps = Variable(torch.FloatTensor(self.latent_size).normal_()).to(device)
        std = log_var.mul(0.5).exp_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        return self.decode(z)
