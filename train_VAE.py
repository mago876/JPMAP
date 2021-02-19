import os
import json
import argparse
import torch
from utils import *
from vae import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--network-structure', type=str, default='VAEfc')  # VAEfc, ConvVAE
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--dvae-sigma', type=int, default=0)  # Noise std in [0,255]
parser.add_argument('--latent-dim', type=int, default=12)
parser.add_argument('--gpu', action='store_true', default=True)
parser.add_argument('--kl-annealing', action='store_true', default=False)
parser.add_argument('--output-path', type=str, default='./pretrained_models')
parser.add_argument('--exp-name', type=str, default='VAEfc_MNIST_01')


args = parser.parse_args()
args.cuda = args.gpu and torch.cuda.is_available()

# Initial setup
torch.manual_seed(1.)
device = torch.device("cuda" if args.cuda else "cpu")
model_path = os.path.join(args.output_path, args.exp_name)
results_path = os.path.join(model_path, 'results')
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load datasets
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader, test_loader = load_dataset(args.dataset, args.batch_size, kwargs)

# Vae training
if args.network_structure == 'VAEfc':
    args.x_shape = train_loader.dataset[0][0].shape
    args.enc_hidden_dims = [512, 512]
    vae = VAEfc(args.x_shape, args.latent_dim, args.enc_hidden_dims)
    vae.fit((train_loader, test_loader), args.epochs, args.lr, args.dvae_sigma, device=device, kl_annealing=args.kl_annealing, path=results_path)

elif args.network_structure == 'ConvVAE':
    args.x_shape = train_loader.dataset[0][0].shape
    args.n_channels = [64,128,256,512]
    vae = ConvVAE(args.x_shape, args.latent_dim, args.n_channels)
    vae.fit((train_loader, test_loader), args.epochs, args.lr, args.dvae_sigma, device=device, kl_annealing=args.kl_annealing, path=results_path)

# Save model
save_vae_model(model_path, vae, args)