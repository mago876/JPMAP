import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

from time import time


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

    def posterior_sample(self, mu_z, logsd_z):
        sd_z = torch.exp(logsd_z)
        eps = torch.randn_like(sd_z)
        return mu_z + eps*sd_z

    def forward(self, x):
        mu_z, logsd_z = self.encoder(x)
        z = self.posterior_sample(mu_z, logsd_z)
        mu_x, gamma_x = self.decoder(z)
        return mu_x, gamma_x, mu_z, logsd_z

    def compute_losses(self, x, mu_x, gamma_x, mu_z, logsd_z):
        BATCH_SIZE = x.size(0)
        HALF_LOG_TWO_PI = 0.91894
        sd_z = torch.exp(logsd_z)
        
        gen_loss = torch.sum(0.5*((x - mu_x)/gamma_x).pow(2) + gamma_x.log() + HALF_LOG_TWO_PI) / BATCH_SIZE
        kl_loss = 0.5*torch.sum(mu_z.pow(2) + sd_z.pow(2) - 1 - 2*logsd_z) / BATCH_SIZE
        
        return gen_loss, kl_loss

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def fit(self, dataloaders, n_epochs=10, lr=1e-3, dvae_sigma=0, device='cuda', kl_annealing=False, nprint=1, path='.'):

        tic = time()

        self.to(device)
        train_loader, test_loader = dataloaders
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Save train and test loss values
        train_gen_losses = []
        train_kl_losses = []
        test_gen_losses = []
        test_kl_losses = []

        for epoch in range(n_epochs):  # loop over the dataset multiple times

            train_loss = 0.0
            self.train()

            # KL annealing
            if kl_annealing:
                beta = np.minimum(1., 3*epoch / n_epochs)
            else:
                beta = 1.

            print('beta (kl_annealing) =', beta)

            # Training steps
            for i, (x,_) in enumerate(train_loader, 0):

                # Load data on device
                x = x.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                if dvae_sigma: # DenoisingVAE
                    x_noisy = x + dvae_sigma/255*torch.randn(*x.shape, device=device)
                    mu_x, gamma_x, mu_z, logsd_z = self.forward(x_noisy)
                else:
                    mu_x, gamma_x, mu_z, logsd_z = self.forward(x)

                gen_loss, kl_loss = self.compute_losses(x, mu_x, gamma_x, mu_z, logsd_z)
                loss = gen_loss + beta*kl_loss
                train_gen_losses.append(float(gen_loss))
                train_kl_losses.append(float(kl_loss))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader.dataset)

            # Save losses
            with open(os.path.join(path, 'training_losses.json'), 'w') as outfile:
                json.dump([train_gen_losses, train_kl_losses], outfile)

            # Print statistics every 'nprint' epochs
            if epoch % nprint == nprint-1:

                # Model evaluation (test)
                with torch.no_grad():
                    self.eval()

                    # Test loss
                    test_loss = 0
                    for i, (data, _) in enumerate(test_loader):
                        data = data.to(device)
                        mu_x, gamma_x, mu_z, logsd_z = self.forward(data)
                        
                        gen_loss, kl_loss = self.compute_losses(data, mu_x, gamma_x, mu_z, logsd_z)
                        test_loss += (gen_loss + kl_loss).item()

                        test_gen_losses.append(float(gen_loss))
                        test_kl_losses.append(float(kl_loss))

                        # Reconstructions
                        if i == 0:
                            n = min(data.size(0), 8)
                            comparison = torch.cat([data[:n], mu_x[:n]])
                            save_image(comparison.cpu(), os.path.join(path,'reconstruction_' + str(epoch) + '.png'), nrow=n)

                    test_loss /= len(test_loader.dataset)

                    print('[Epoch %d / %d]  Train loss = %.5f  |  Test loss = %.5f  |  gamma_x = %.4f  |  %.2f min'
                                           % (epoch+1, n_epochs, train_loss, test_loss, gamma_x.item(), (time()-tic)/60))

                    # Random samples
                    sample = torch.randn(64, self.latent_dim).to(device)
                    sample = self.decoder(sample)[0].cpu()
                    save_image(sample, os.path.join(path,'sample_' + str(epoch) + '.png'))

            # Save losses
            with open(os.path.join(path, 'test_losses.json'), 'w') as outfile:
                json.dump([test_gen_losses, test_kl_losses], outfile)

        toc = time()
        print('Training finished - Elapsed time: %.2f sec' % (toc-tic))
        

# Fully connected VAE
class VAEfc(VAE):
    def __init__(self, x_shape, latent_dim, enc_hidden_dims):
        super(VAEfc, self).__init__()

        self.x_shape = x_shape
        self.input_dim = np.prod(x_shape)
        self.latent_dim = latent_dim
        self.num_hidden_layers = len(enc_hidden_dims)

        self.encoder_layers = nn.ModuleList([nn.Linear(self.input_dim, enc_hidden_dims[0])])
        self.encoder_layers.extend([nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i+1]) for i in range(self.num_hidden_layers-1)])
        self.encoder_layers.append(nn.Linear(enc_hidden_dims[-1], self.latent_dim))  # mu_z
        self.encoder_layers.append(nn.Linear(enc_hidden_dims[-1], self.latent_dim))  # logsd_z

        self.decoder_layers = nn.ModuleList([nn.Linear(self.latent_dim, enc_hidden_dims[-1])])
        self.decoder_layers.extend([nn.Linear(enc_hidden_dims[i], enc_hidden_dims[i-1]) for i in range(self.num_hidden_layers-1,0,-1)])
        self.decoder_layers.append(nn.Linear(enc_hidden_dims[0], self.input_dim))  # mu_x
        self.gamma_x = nn.Parameter(torch.ones(1))  # gamma_x

    def encoder(self, x):
        #h = F.relu(self.encoder_layers[0](x.view(-1,self.input_dim)))  # ReLU
        h = F.elu(self.encoder_layers[0](x.view(-1,self.input_dim)))  # ELU
        #h = F.gelu(self.encoder_layers[0](x.view(-1,self.input_dim)))  # GELU
        for i in range(1,self.num_hidden_layers):
            #h = F.relu(self.encoder_layers[i](h))  # ReLU
            h = F.elu(self.encoder_layers[i](h))  # ELU
            #h = F.gelu(self.encoder_layers[i](h))  # GELU
        mu_z = self.encoder_layers[-2](h)
        logsd_z = self.encoder_layers[-1](h)
        return mu_z, logsd_z

    def decoder(self, z):
        #h = F.relu(self.decoder_layers[0](z))  # ReLU
        h = F.elu(self.decoder_layers[0](z))  # ELU
        #h = F.gelu(self.decoder_layers[0](z))  # GELU
        for i in range(1,self.num_hidden_layers):
            #h = F.relu(self.decoder_layers[i](h))  # ReLU
            h = F.elu(self.decoder_layers[i](h))  # ELU
            #h = F.gelu(self.decoder_layers[i](h))  # GELU
        mu_x = torch.sigmoid(self.decoder_layers[-1](h))
        return mu_x.view(-1,*self.x_shape), self.gamma_x


# Convolutional VAE
class ConvVAE(VAE):
    def __init__(self, x_shape, latent_dim, n_channels, kernel_size=5, activation='elu'):
        # n_channels must be a list of length 4 (e.g. [64,128,256,512])
        super(ConvVAE, self).__init__()
        self.x_shape = x_shape
        self.im_size = x_shape[1]*x_shape[2]  # [C,H,W] format
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.n_channels = n_channels  # For convolutional layers
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu
        elif activation == 'gelu':
            self.activation = F.gelu

        # Encoder layers
        self.conv_enc1 = nn.Conv2d(x_shape[0], self.n_channels[0], kernel_size, stride=1, padding=kernel_size//2)
        self.batchnorm_enc1 = nn.BatchNorm2d(self.n_channels[0])
        self.conv_enc2 = nn.Conv2d(self.n_channels[0], self.n_channels[1], kernel_size, stride=2, padding=kernel_size//2)
        self.batchnorm_enc2 = nn.BatchNorm2d(self.n_channels[1])
        self.conv_enc3 = nn.Conv2d(self.n_channels[1], self.n_channels[2], kernel_size, stride=2, padding=kernel_size//2)
        self.batchnorm_enc3 = nn.BatchNorm2d(self.n_channels[2])
        self.conv_enc4 = nn.Conv2d(self.n_channels[2], self.n_channels[3], kernel_size, stride=2, padding=kernel_size//2)
        self.batchnorm_enc4 = nn.BatchNorm2d(self.n_channels[3])
        self.fc_enc = nn.Linear(self.im_size*self.n_channels[3]//64, 1024)
        self.fc_mu_z    = nn.Linear(1024, latent_dim)  # mu_z
        self.fc_logsd_z = nn.Linear(1024, latent_dim)  # logsd_z

        # Decoder layers
        self.fc_dec1 = nn.Linear(latent_dim, 1024)
        self.fc_dec2 = nn.Linear(1024, self.im_size*self.n_channels[3]//64)
        self.convtrans_dec1 = nn.ConvTranspose2d(self.n_channels[3], self.n_channels[2], kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        self.batchnorm_dec1 = nn.BatchNorm2d(self.n_channels[2])
        self.convtrans_dec2 = nn.ConvTranspose2d(self.n_channels[2], self.n_channels[1], kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        self.batchnorm_dec2 = nn.BatchNorm2d(self.n_channels[1])
        self.convtrans_dec3 = nn.ConvTranspose2d(self.n_channels[1], self.n_channels[0], kernel_size, stride=2, padding=kernel_size//2, output_padding=1)
        self.batchnorm_dec3 = nn.BatchNorm2d(self.n_channels[0])
        self.convtrans_dec4 = nn.ConvTranspose2d(self.n_channels[0], x_shape[0],  kernel_size, stride=1, padding=kernel_size//2)  # mu_x
        self.batchnorm_dec4 = nn.BatchNorm2d(self.x_shape[0])
        self.gamma_x = nn.Parameter(torch.ones(1))  # gamma_x
    
    def encoder(self, x):
        h = self.activation(self.batchnorm_enc1(self.conv_enc1(x)))
        h = self.activation(self.batchnorm_enc2(self.conv_enc2(h)))
        h = self.activation(self.batchnorm_enc3(self.conv_enc3(h)))
        h = self.activation(self.batchnorm_enc4(self.conv_enc4(h)))
        h = self.activation(self.fc_enc(h.view(-1,self.im_size*self.n_channels[3]//64)))
        mu_z = self.fc_mu_z(h)
        logsd_z = self.fc_logsd_z(h)
        return mu_z, logsd_z

    def decoder(self, z):
        h = self.activation(self.fc_dec1(z))
        h = self.activation(self.fc_dec2(h)).view(-1, self.n_channels[3], self.x_shape[1]//8, self.x_shape[2]//8)  # [C,H,W] format
        h = self.activation(self.batchnorm_dec1(self.convtrans_dec1(h)))
        h = self.activation(self.batchnorm_dec2(self.convtrans_dec2(h)))
        h = self.activation(self.batchnorm_dec3(self.convtrans_dec3(h)))
        mu_x = torch.sigmoid(self.batchnorm_dec4(self.convtrans_dec4(h)))
        return mu_x, self.gamma_x