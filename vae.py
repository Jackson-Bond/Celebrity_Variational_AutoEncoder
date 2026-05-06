# Bond, Jackson
# 1002_032_254
# 2026_03_29
# Assignment_04_03

import torch
import torch.nn as nn
import torch.nn.functional as F


def vaeLoss(reconstructedX, x, mu, logvar, beta=4.0):
    # reconstruction loss
    reconstructionLoss = F.binary_cross_entropy(reconstructedX, x, reduction='sum')
    # KL divergence, keeps the latent space organized and smooth
    klLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (reconstructionLoss + beta * klLoss) / x.size(0)

class VAE(nn.Module):
    def __init__(self, latentDim=6, imageChannels=3):
        super().__init__()
        self.latentDim = latentDim

        # encoder, compress 64x64 image down to latent space
        self.encoder = nn.Sequential(
            # 3 x 64 x 64 -> 32 x 32 x 32
            nn.Conv2d(imageChannels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )

        flattenSize = 256 * 4 * 4

        # two separate heads to produce mean and log variance for reparameterization
        self.fcMu = nn.Linear(flattenSize, latentDim)
        self.fcLogvar = nn.Linear(flattenSize, latentDim)

        # decoder, expand latent vector back to 64x64 image
        self.fcDecode = nn.Linear(latentDim, flattenSize)

        self.decoder = nn.Sequential(
            # 256 x 4 x 4 -> 128 x 8 x 8
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, imageChannels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fcMu(h)
        logvar = self.fcLogvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # sample from the distribution during training, use mean at inference
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h = self.fcDecode(z)
        h = h.view(h.size(0), 256, 4, 4)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar