import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, zdim):
        super(Encoder, self).__init__()
        self.zdim = zdim

        # construct the body
        body_list = []
        bl = nn.Linear(784, self.zdim * 2)
        body_list.append(bl)
        self.body = nn.Sequential(*body_list)

    def forward(self, x):
        scores = self.body(x)
        mu, logvar = torch.chunk(scores, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        
        qz = torch.distributions.Normal(mu, std)
        z = qz.rsample()
        logqz = qz.log_prob(z)
        
        return z, logqz

class Decoder(nn.Module):
    def __init__(self, zdim):
        super(Decoder, self).__init__()
        self.zdim = zdim

        # construct the body
        body_list = []
        bl = nn.Linear(self.zdim, 784)
        body_list.append(bl)
        self.body = nn.Sequential(*body_list)

    def forward(self, z):
        xhat = torch.sigmoid(self.body(z))
        return xhat

class VAE(nn.Module):
    def __init__(self, zdim):
        super(VAE, self).__init__()
        self.zdim = zdim
        self.encoder = Encoder(zdim)
        self.decoder = Decoder(zdim)

    def forward(self, x):
        z, logqz = self.encoder(x.view(-1, 784))
        xhat = self.decoder(z)
        
        return xhat,z,logqz

    def sample(self, num_samples):
        with torch.no_grad():
            z = torch.randn(num_samples, self.zdim).to(device)
            samples = self.decoder(z)
        return samples

    def loss_function(self, x, x_hat, mu, logvar):
        BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD


# Define hyperparameters
batch_size = 128
input_size = 784
hidden_size = 400
latent_size = 20
lr = 0.001
num_epochs = 20

# Create dataloader
train_dataset = MNIST(root='./data', train=True, download=True, transform=ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Create model and optimizer
model = VAE(latent_size)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
model.train()
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_dataloader):
        x = x.view(-1, input_size)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(x)
        loss = model.loss_function(x, x_hat, mu, logvar)
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
    x_hat, _, _ = model(x_sample)

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
    plt.imshow(x_hat[i].view(28, 28), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()