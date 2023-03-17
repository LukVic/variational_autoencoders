import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, zdim):
        super(Encoder, self).__init__()
        self.zdim = zdim

        # construct the body
        body_list = []
        bl = nn.Linear(784, 512)
        body_list.append(bl)
        body_list.append(nn.ReLU())
        bl = nn.Linear(512, 256)
        body_list.append(bl)
        body_list.append(nn.ReLU())
        bl = nn.Linear(256, self.zdim * 2) 
        body_list.append(bl)
        self.body = nn.Sequential(*body_list)

    def forward(self, x):
        scores = self.body(x)
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        sigma = torch.exp(sigma)

        # Sample from the distribution using reparameterization trick
        qz = torch.distributions.Normal(mu, sigma)
        z = qz.rsample()

        return z, mu, sigma


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, zdim):
        super(Decoder, self).__init__()
        self.zdim = zdim

        # construct the body
        body_list = []
        bl = nn.Linear(self.zdim, 256)
        body_list.append(bl)
        body_list.append(nn.ReLU())
        bl = nn.Linear(256, 512)
        body_list.append(bl)
        body_list.append(nn.ReLU())
        bl = nn.Linear(512, 784)
        body_list.append(bl)
        body_list.append(nn.Sigmoid())
        self.body = nn.Sequential(*body_list)

    def forward(self, x):
        x_hat = self.body(x)
        return x_hat


# Define the Variational Autoencoder
class VAE(nn.Module):
    def __init__(self, zdim):
        super(VAE, self).__init__()
        self.encoder = Encoder(zdim)
        self.decoder = Decoder(zdim)

    def forward(self, x):
        z, mu, sigma = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma


# Set device to GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define hyperparameters
epochs = 20
batch_size = 128
lr = 1e-3
zdim = 10

# Load the dataset
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize the model and optimizer
model = VAE(zdim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define the loss function
def loss_function(x_hat, x, mu, sigma):
    # Reconstruction loss
    recon_loss = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
    # KL divergence
    qz = torch.distributions.Normal(mu, sigma)
    pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
    kl_div = torch.distributions.kl_divergence(qz, pz).sum()
    # Total loss
    loss = recon_loss + kl_div
    return loss

# Train the model
model.train()
for epoch in range(epochs):
    train_loss = 0.0
    for x, _ in tqdm(trainloader):
        x = x.to(device)
        optimizer.zero_grad()
        x_hat, mu, sigma = model(x)
        loss = loss_function(x_hat, x, mu, sigma)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch+1}, loss = {train_loss / len(trainset):.3f}")

# Test the model by
