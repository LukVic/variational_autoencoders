import torch
import torchvision
from torch import nn
import torchvision.datasets as datasets


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
        mu, sigma = torch.split(scores, self.zdim, dim=1)
        sigma = torch.exp(sigma)
 
        return mu, sigma

class Decoder(nn.Module):
    def __init__(self, zdim):
        super(Decoder, self).__init__()
        # construct the body
        body_list = []
        bl = nn.Linear(zdim, 784)
        body_list.append(bl)
        self.body = nn.Sequential(*body_list)
 
    def forward(self, x):
        mu = self.body(x)
 
        return mu     


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
training_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=16, shuffle=True)
device = torch.device('cuda')
enc = Encoder()
enc.train()
z_um, z_sigma = enc.eval()


# initialise a tensor of normal distributions from tensors z_mu, z_sigma
qz = torch.distributions.Normal(z_mu, z_sigma)
 
# compute log-probabilities for a tensor z
logz = qz.log_prob(z)
 
# sample from the distributions with re-parametrisation
zs = qz.rsample()
 
# compute KL divergence for two tensors with probability distributions
kl_div = torch.distributions.kl_divergence(qz, pz)