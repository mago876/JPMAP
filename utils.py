import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from vae import *


def load_dataset(dataset, batch_size, kwargs):

    if dataset == 'mnist':
        train_loader = DataLoader(
                    datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)

    elif dataset == 'fashionmnist':
        train_loader = DataLoader(
                    datasets.FashionMNIST('data', train=True, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.FashionMNIST('data', train=False, download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)

    elif dataset == 'svhn':
        train_loader = DataLoader(
                    datasets.SVHN('data', split='train', download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.SVHN('data', split='test', download=True, transform=transforms.ToTensor()),
                    batch_size=batch_size, shuffle=True, **kwargs)

    elif dataset == 'celeba':
        ### There is a issue with CelebA on PyTorch code, see https://github.com/pytorch/vision/issues/2262 ###
        transform = transforms.Compose([transforms.CenterCrop(128), transforms.Resize(64), transforms.ToTensor()])
        train_loader = DataLoader(
                    datasets.CelebA('data', split='train', download=True, transform=transform),
                    batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(
                    datasets.CelebA('data', split='test', download=True, transform=transform),
                    batch_size=batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader


def save_vae_model(path, vae, args):

    torch.save(vae.state_dict(), os.path.join(path, 'model'))
    with open(os.path.join(path, 'training_args.json'), 'w') as outfile:
        json.dump(vars(args), outfile)


def load_vae_model(path, device='cuda'):

    with open(os.path.join(path, 'training_args.json')) as json_file:
        args = json.load(json_file)

    if args['network_structure'] == 'VAEfc':
        vae = VAEfc(args['x_shape'], args['latent_dim'], args['enc_hidden_dims'])

    elif args['network_structure'] == 'ConvVAE':
        vae = ConvVAE(args['x_shape'], args['latent_dim'], args['n_channels'])

    vae.load_state_dict(torch.load(os.path.join(path, 'model')))

    return vae.to(device)