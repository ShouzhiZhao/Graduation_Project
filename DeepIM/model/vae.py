import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, latent_dim)
        
        self.bn = nn.BatchNorm1d(num_features=latent_dim)
        
    def forward(self, x):
        h_ = F.relu(self.FC_input(x))
        h_ = F.relu(self.FC_input2(h_))
        h_ = F.relu(self.FC_input2(h_))
        output = self.FC_output(h_)
        return output


class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, latent_dim)
        self.FC_hidden_1 = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden_2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        #self.prelu = nn.PReLU()
        
    def forward(self, x):
        h = F.relu(self.FC_input(x))
        h = F.relu(self.FC_hidden_1(h))
        h = F.relu(self.FC_hidden_2(h))
        # x_hat = self.FC_output(h)
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAEModel(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAEModel, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        std = torch.exp(0.5*var) # standard deviation
        epsilon = torch.randn_like(var)
        return mean + std*epsilon

    def forward(self, x, adj=None):
        if adj != None:
            z = self.Encoder(x, adj)
        else:
            z = self.Encoder(x)
        # z = mean + log_var # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat