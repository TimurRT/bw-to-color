import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 4, stride=2, padding=1)  # 32x16x16
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1) # 64x8x8
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1) # 128x4x4
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128*4*4, latent_dim)
        self.fc_logvar = nn.Linear(128*4*4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128*4*4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1) # 64x8x8
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 32x16x16
        self.deconv3 = nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1)   # 2x32x32

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.deconv3(x)) 
        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        ab = self.decoder(z)
        return ab, mu, logvar
