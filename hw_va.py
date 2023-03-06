import torch
import torchvision
from torch import nn
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,random_split

class DataLoad():

    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self):
        data_dir = self.data_dir

        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

        train_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

        train_dataset.transform = train_transform
        test_dataset.transform = test_transform

        m=len(train_dataset)

        train_data, val_data = random_split(train_dataset, [int(m-m*0.2), int(m*0.2)])
        batch_size=256

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,shuffle=True)

        return train_loader, valid_loader, test_loader


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

        # initialise a tensor of normal distributions from tensors z_mu, z_sigma
        qz = torch.distributions.Normal(mu, sigma)
        # sample from the distributions with re-parametrisation
        zs = qz.rsample()
 
        return zs

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

class VariationalAutoencoder(nn.Module):
    def __init__(self, zdim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(zdim)
        self.decoder = Decoder(zdim)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        return self.decoder()
    
    def train_step(self):
        pass



# mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
# training_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=16, shuffle=True)
# device = torch.device('cuda')

dl = DataLoad('./data')
tr_data, val_data, tst_data = dl.load_data()

zdims = 100


vae = VariationalAutoencoder(zdim=zdims)
stepsize = 1e-3

optimizer = torch.optim.Adam(vae.parameters(), lr=stepsize, weight_decay=1e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

vae.to(device)




# # # initialise a tensor of normal distributions from tensors z_mu, z_sigma
# qz = torch.distributions.Normal(z_mu, z_sigma)
 
# # compute log-probabilities for a tensor z
# logz = qz.log_prob(z)
 
# # sample from the distributions with re-parametrisation
# zs = qz.rsample()
 
# # compute KL divergence for two tensors with probability distributions
# kl_div = torch.distributions.kl_divergence(qz, pz)