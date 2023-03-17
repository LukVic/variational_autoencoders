import torch
import torchvision
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader,random_split

PATH = "/home/lucas/Documents/KYR/msc_thesis/hw/data/"

class DataLoad():

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        self.test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)

    def load_data(self):

        train_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

        self.train_dataset.transform = train_transform
        self.test_dataset.transform = test_transform

        m=len(self.train_dataset)

        train_data, val_data = random_split(self.train_dataset, [int(m-m*0.2), int(m*0.2)])
        batch_size=256

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        valid_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size,shuffle=True)

        return train_loader, valid_loader, test_loader
    


class Encoder(nn.Module):
    def __init__(self, zdim):
        super(Encoder, self).__init__()
        self.zdim = zdim

        # construct the body
        body_list = []
        bl = nn.Linear(784, self.zdim * 2) 
        body_list.append(bl)
        il1 = nn.ReLU()
        body_list.append(il1)
        #il2 = nn.Linear(self.zdim * 2, self.zdim * 2)
        #body_list.append(il2)
        # il3 = nn.ReLU()
        # body_list.append(il3)
        #il2 = nn.Linear(self.zdim * 2, self.zdim * 2)
        #body_list.append(il2)
        il2 = nn.Linear(self.zdim * 2, self.zdim * 2)
        body_list.append(il2)
        self.body = nn.Sequential(*body_list)
 
    def forward(self, x):
        x = x.to(device)
        #print(len(x))
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
        super().__init__()
        # construct the body
        body_list = []
        bl = nn.Linear(zdim, 784)
        body_list.append(bl)
        # il1 = nn.ReLU()
        # body_list.append(il1)
        # il2 = nn.Linear(self.zdim, self.zdim)
        # body_list.append(il2)
        # il3 = nn.ReLU()
        # body_list.append(il3)
        # il4 = nn.Linear(self.zdim,self.zdim),
        # body_list.append(il4)
        il5 = nn.Sigmoid()
        body_list.append(il5)
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

    def recon(self, x):
        x_hat = vae(x)
        x_hat = x_hat.to(device)
        log_scale = nn.Parameter(torch.Tensor([0.0]))
        scale = torch.exp(log_scale)
        scale = scale.to(device)
        pxz = torch.distributions.Normal(x_hat, scale)
        log_pxz = pxz.log_prob(x)
        E_log_pxz = log_pxz.sum()

        return E_log_pxz

    def train_step(self, vae, device, dataloader, optimizer):
        
        vae.train()
        train_loss = 0.0

        for x, _ in dataloader:
            x = x.to(device)
            elbo = vae.kl - self.recon(x.view(-1,784))
            loss = elbo.mean()

            optimizer.zero_grad()
            loss.backward(retain_graph = True)
            optimizer.step()
            # Print batch loss
            print('\t partial train loss (single batch): %f' % (loss.item()))
            train_loss+=loss.item()

        return train_loss / len(dataloader.dataset)
    
    def test_step(self, vae, device, dataloader):
        
        vae.eval()
        val_loss = 0.0
        with torch.no_grad(): # No need to track the gradients
            for x, _ in dataloader:
                
                x = x.to(device)
                # Decode data
                elbo = vae.kl - self.recon(x.view(-1,784))
                loss = elbo.mean()
                val_loss += loss.item()

        return val_loss / len(dataloader.dataset)
    

    def plot_ae_outputs(self, encoder,decoder,test_dataset,n=10):
        plt.figure(figsize=(16,4.5))
        targets = test_dataset.targets.numpy()
        t_idx = {i:np.where(targets==i)[0][0] for i in range(n)}
        for i in range(n):
            ax = plt.subplot(2,n,i+1)
            img = test_dataset[t_idx[i]][0].unsqueeze(0).to(device)
            encoder.eval()
            decoder.eval()
            with torch.no_grad():
                rec_img  = decoder(encoder(img.view(-1,784)))
            plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Original images')
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  
            if i == n//2:
                ax.set_title('Reconstructed images')
        plt.show()  

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
data_loader = DataLoad(PATH)
train_loader, valid_loader, test_loader = data_loader.load_data()

num_epochs = 50

for epoch in range(num_epochs):
   train_loss = vae.train_step(vae,device,train_loader,optimizer)
   val_loss = vae.test_step(vae,device,valid_loader)
   print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
  # vae.plot_ae_outputs(vae.encoder,vae.decoder,data_loader.test_dataset, n=10)




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