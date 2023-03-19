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
            nn.Linear(784, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, self.zdim * 2)
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
            nn.Linear(self.zdim, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 120),
            nn.ReLU(),
            nn.Linear(120, 784),
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
        std = torch.exp(sigma)  
        qz = torch.distributions.Normal(mu, std)
        z = qz.rsample()
        logqz = qz.log_prob(z)
        xhat = self.decoder(z)
        
        return xhat,z,logqz,qz,std

    def count_params(self):
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

    def recon(self, x_hat, x):
        
        #log_scale = nn.Parameter(torch.Tensor([0.0])).to('cuda')

        # scale = torch.exp(log_scale)
        # scale = torch.exp(0.5 * logvar)
        # #pxz = torch.distributions.Normal(x_hat, scale)
        # #log_pxz = pxz.log_prob(x.to('cuda'))
        # #E_log_pxz = -log_pxz.sum()

        pxz = Normal(x_hat, torch.ones_like(x_hat))
        E_log_pxz = -pxz.log_prob(x.to('cuda')).sum(dim=1)
        return E_log_pxz
        # return E_log_pxz

    def loss_function(self, x, x_hat, mu, logvar, qz, sigma, i):
        BCE = self.recon(x_hat, x)
        #eps = 0.00000001
        x_hat = x_hat.to('cuda')
        x = x.to('cuda')
        #BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')
        #BCE = -torch.sum(x * torch.log(x_hat + eps) + (1 - x) * torch.log(1 - x_hat + eps))
        
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        #KLD = torch.distributions.kl_divergence(qz, pz).sum()
        KLD = torch.distributions.kl_divergence(qz, pz).sum(dim=1)
        # if i == 0:
        #     beta = 0.0
        # elif i == 1:
        #     beta = 0.1
        # else: beta = 0.3
        #if i % 2 == 0: beta = 0.2
        #else: beta = 0.7

        beta = 0.1

        return torch.mean(BCE + beta*KLD)
    

# Define hyperparameters
batch_size = 128
input_size = 784
latent_size = 20
lr = 0.001
num_epochs = 5
elbo_history = []

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
        loss = model.loss_function(x, x_hat, mu, logvar, qz, std, i)
        loss.backward()
        optimizer.step()
        elbo_history.append(loss.item())
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.item()))

# Generate some samples and their reconstructions
model.eval()
with torch.no_grad():
    x_sample, _ = next(iter(train_dataloader))
    x_sample = x_sample.view(-1, input_size)
    x_hat, _, _,_,_ = model(x_sample)

# Visualize the results
num_samples = 16
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



with torch.no_grad():
        kl_divs = []
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.to('cuda')
            recon_batch, mu, logvar, qz, std = model(data)
            z = qz.rsample()
            pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            kl_div = torch.distributions.kl_divergence(qz, pz)
            kl_divs.append(kl_div)
            break
        
        kl_divs = torch.cat(kl_divs, dim=0)
        kl_divs_mean = kl_divs.mean(dim=0)
        
        # plot histogram of averaged kl divergences for each latent space component
        kl_divs_mean = kl_divs_mean.cpu().numpy()
        plt.figure()
        plt.hist(kl_divs_mean, bins=20)
        plt.show()
        
        qz_var = kl_divs_mean.var().item()
        qz_mean = kl_divs_mean.mean().item()
        qz_var / qz_mean




plt.figure()
plt.plot(elbo_history)
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Total Loss vs. Epoch')
plt.show()

print("Number of VAE params: {0}".format(model.count_params()))