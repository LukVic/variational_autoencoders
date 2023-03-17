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
        #qz = torch.distributions.Normal(mu, sigma)
        #p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        # sample from the distributions with re-parametrisation
        #zs = qz.rsample()
 
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

class VariationalAutoencoder(nn.Module):
    def __init__(self, zdim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(zdim)
        self.decoder = Decoder(zdim)
        self.kl = 0

    def forward(self, x):
        x = x.to(device)
        mu, sigma = self.encoder(x)
        qz = torch.distributions.Normal(mu, sigma)
        pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        zs = qz.rsample()
        self.kl = torch.distributions.kl_divergence(qz, pz)
        return self.decoder(zs)

    def recon(x):
        x_hat = vae(x)
        log_scale = nn.Parameter(torch.Tensor([0.0]))
        scale = torch.exp(log_scale)
        pxz = torch.distributions.Normal(x_hat, scale)
        log_pxz = pxz.log_prob(x)
        E_log_pxz = log_pxz.sum(dim=(1, 2, 3))

        return E_log_pxz

    def train_step(self, vae, device, dataloader, optimizer):
        
        vae.train()
        train_loss = 0.0

        for x, _ in dataloader:
            x = x.to(device)

            elbo = vae.kl - self.recon(x)
            loss = elbo.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Print batch loss
            print('\t partial train loss (single batch): %f' % (loss.item()))
            train_loss+=loss.item()

        return train_loss / len(dataloader.dataset)
    
    def test_epoch(self, vae, device, dataloader):
        
        vae.eval()
        val_loss = 0.0
        with torch.no_grad(): # No need to track the gradients
            for x, _ in dataloader:
                
                x = x.to(device)
                # Decode data
                elbo = vae.kl - self.recon(x)
                loss = elbo.mean()
                val_loss += loss.item()

        return val_loss / len(dataloader.dataset)
    

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


    # def forward(self, x):
    #     x = x.to(device)
    #     mu, sigma = self.encoder(x)
    #     qz = torch.distributions.Normal(mu, sigma)
    #     pz = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

    #     zs = qz.rsample()

    #     self.kl_div = torch.distributions.kl_divergence(qz, pz)
    
    #     return self.decoder(zs)