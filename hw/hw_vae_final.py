import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, zdim):
        super(Encoder, self).__init__()
        self.zdim = zdim

        self.body = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.zdim * 2)
        )

    def forward(self, x):
        scores = self.body(x.to('cuda'))
        mu, sigma = torch.chunk(scores, 2, dim=1)
        
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, zdim):
        super(Decoder, self).__init__()
        self.zdim = zdim

        self.body = nn.Sequential(
            nn.Linear(self.zdim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, z):
        xhat = self.body(z)
        return xhat

class VAE(nn.Module):
    def __init__(self, zdim):
        super(VAE, self).__init__()
        self.device = 'cuda:0'
        self.zdim = zdim
        self.encoder = Encoder(zdim).to('cuda')
        self.decoder = Decoder(zdim).to('cuda')

    def forward(self, x):
        mu, sigma = self.encoder(x.view(-1, 784))
        std = torch.exp(0.5 * sigma)  
        qz = torch.distributions.Normal(mu, std)
        z = qz.rsample()
        logqz = qz.log_prob(z)
        xhat = self.decoder(z)
        
        return xhat,z,logqz,qz,std

    def recon(self, x_hat, x):
        log_scale = nn.Parameter(torch.Tensor([0.0])).to('cuda')
        scale = torch.exp(log_scale)
        pxz = torch.distributions.Normal(x_hat, scale)
        log_pxz = pxz.log_prob(x.to('cuda'))
        E_log_pxz = -log_pxz.sum()

        return E_log_pxz

    def loss_function(self, x, x_hat, mu, logvar, qz, sigma):
        BCE = self.recon(x_hat, x)
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        KLD = torch.distributions.kl_divergence(qz, pz).sum()
        
        return BCE + KLD
    

# Define hyperparameters
batch_size = 32
input_size = 784
latent_size = 20
lr = 0.001
num_epochs = 15

# Create dataloader
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create model and optimizer
model = VAE(latent_size).to('cuda')
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
model.train()
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_dataloader):
        x = x.view(-1, input_size)
        optimizer.zero_grad()
        x_hat, mu, logvar, qz, std = model(x)
        loss = model.loss_function(x, x_hat, mu, logvar, qz, std)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

# Generate some samples and their reconstructions
# Generate some samples and their reconstructions
model.eval()
with torch.no_grad():
    x_sample, _ = next(iter(train_dataloader))
    x_sample = x_sample.view(-1, input_size)
    x_hat, _, _,_,_ = model(x_sample)

# Visualize the results
num_samples = 5
plt.figure(figsize=(10, 4))

# Show the original images
for i in range(num_samples):
    plt.subplot(2, num_samples, i + 1)
    plt.imshow(x_sample[i].view(28, 28), cmap='gray')
    plt.axis('off')

# Show the reconstructed images
for i in range(num_samples):
    plt.subplot(2, num_samples, num_samples + i + 1)
    plt.imshow(x_hat[i].to('cpu').view(28, 28), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()