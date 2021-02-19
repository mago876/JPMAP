import os
import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image

from time import time


def z_step(xk, vae, beta, zinit=None, max_iters=1000, tol=1e-4, lr_adam = 0.01, verbose=False):
    # OUTPUT:
    #        z step:  argmin_z (1/2)*||z||^2 + (beta/2)*||xk - Dec(z)||^2

    if verbose: 
        print('Running z_step...')
        print('lr_adam = ', lr_adam)

    zdim = vae.latent_dim

    if zinit is None:
        with torch.no_grad():
            zinit = vae.encoder(xk[None,:])[0].data  # Initialize at mu_E(xk)

    zk = zinit.detach().clone().requires_grad_(True)

    # Optimizer
    optimizer = optim.Adam([zk], lr=lr_adam)

    convergence = False
    k = 0

    while not convergence and k<max_iters:
        k += 1
        optimizer.zero_grad()

        # lossF = (1/2)*||z||^2 + (beta/2)*||xk - Dec(z)||^2
        Dxz = 0.5*torch.sum((xk - vae.decoder(zk)[0]).pow(2))
        loss = (0.5/beta)*torch.sum(zk.pow(2)) + Dxz

        loss.backward()
        grad = zk.grad

        if torch.norm(grad)/zdim < tol:
            convergence = True
        else:
            optimizer.step()

    if verbose:
        print('Adam terminated in %d iterations (out of %d) (z-step JPMAP)  |  norm(grad)/zdim at last iteration (z-step JPMAP): %.4f' % (k, max_iters, torch.norm(grad)/zdim))

    return zk.data


# Generic x-step:
def x_step_A(y, x_shape, sigma, A, mu_D, beta, verbose=False):
    # OUTPUT:
    #        x step:  argmin_x (1/2*sigma^2)*||Ax-y||^2 + D(x, z^k)

    if verbose: 
        print('Running x_step_A...')

    mu_D = mu_D.view(-1)
    Amatrix = torch.matmul(A.t(),A)/beta + (sigma**2)*torch.eye(mu_D.nelement(), out=torch.empty_like(A))
    btensor = torch.matmul(A.t(),y)/beta + (sigma**2)*mu_D

    x_new = torch.solve(btensor.view(-1,1), Amatrix)[0]

    return x_new.view(x_shape)#.astype(xtilde.dtype)


# Efficient x-step: Denoising
def x_step_Denoising(y, x_shape, sigma, A, mu_D, beta, verbose=False):

    if verbose: 
        print('Running x_step_Denoising...')
    mu_D = mu_D.view(-1)
    vect = y/beta + (sigma**2)*mu_D
    x_new = vect / (1/beta+sigma**2)

    return x_new.view(x_shape)#.astype(xtilde.dtype)


# Efficient x-step: Missing Pixels
def x_step_MissingPixels(y, x_shape, sigma, A, mu_D, beta, verbose=False):

    if verbose: 
        print('Running x_step_MissingPixels...')
    mask = torch.diag(A).view(-1)
    mu_D = mu_D.view(-1)
    vect = mask*y/beta + (sigma**2)*mu_D
    den = mask/beta + sigma**2
    x_new = vect / den

    return x_new.view(x_shape)#.astype(xtilde.dtype)


def jpmap(y, x_shape, vae, A, x_step, sigma, xtarget=None, max_iters=500, max_iters_inner=500, verbose=True, xinit=None, uzawa=False, device='cuda', save_iters=False, params=None):

    # Setup
    n = x_shape.numel()  # Number of pixels
    zdim = vae.latent_dim
    vae = vae.to(device)

    zinit = torch.zeros(zdim, device=device)
    zk = zinit.requires_grad_(True)

    with torch.no_grad():
        gamma = vae.decoder(zk)[1]
    print('gamma =', gamma.data)

    # A full matrix:
    if xinit is None:
        xk = torch.matmul(A.T,y).view(x_shape)
    else:
        xk = xinit

    # Adam parameters (for z step)
    max_iters_z = 3000

    ## Exponential multiplier method (Uzawa)
    if uzawa:
        beta = 0.1/gamma**2  # beta inicial
        rho = params['rho']/n
        alpha = (params['alpha']/255)**2 * n
    else:
        beta = 1/gamma**2  # Diagonal variance of decoder

    terminate = 0  # Main loop flag
    k = 0  # Iteration counter (outer loop)
    k_inner = 0  # Iteration counter (inner loop)

    x = xk
    z = zk
    J1_prev = np.Inf

    if save_iters:
        xiters_jpmap = np.array(xk.cpu().detach().clone())  # MNIST
        ziters_jpmap = np.array(zk[None,:].cpu().detach().clone())
        beta_k = [beta]
        indices = []
        ind_k = []

    # Main loop
    while not terminate:
        k_inner += 1
        tol = 1/255  # 1/255 corresponds to a MSE of 1 gray level
    
        zE = vae.encoder(xk[None,:])[0].detach()  # Initialize at mu_E(xk)
        mu_D_E = vae.decoder(zE)[0]
        xE = x_step(y, x_shape, sigma, A, mu_D_E, beta)
        J1_E = J1(xE, zE, A, y, sigma, beta, vae)
        
        if J1_E < J1_prev:
            zk = zE
            xk = xE
            mu_D = mu_D_E
            J1_prev = J1_E
            if save_iters:
                ind_k.append(0)

        else:
            zE_GD = z_step(xk, vae, beta, zinit=zE, max_iters=max_iters_z)
            mu_D_E_GD = vae.decoder(zE_GD)[0].detach()
            xE_GD = x_step(y, x_shape, sigma, A, mu_D_E_GD, beta)
            J1_E_GD = J1(xE_GD, zE_GD, A, y, sigma, beta, vae)

            if J1_E_GD < J1_prev:
                zk = zE_GD
                xk = xE_GD
                mu_D = mu_D_E_GD
                J1_prev = J1_E_GD
                if save_iters:
                    ind_k.append(1)

            else:
                zk_GD = z_step(xk, vae, beta, zinit=zk, max_iters=max_iters_z)
                mu_D_k_GD = vae.decoder(zk_GD)[0].detach()
                xk_GD = x_step(y, x_shape, sigma, A, mu_D_k_GD, beta)
                J1_k_GD = J1(xk_GD, zk_GD, A, y, sigma, beta, vae)

                zk = zk_GD
                xk = xk_GD
                mu_D = mu_D_k_GD
                J1_prev = J1_k_GD
                if save_iters:
                    ind_k.append(2)

        ### Convergence criterion
        delta_x = torch.norm(x-xk) / np.sqrt(n) if k_inner!=0 else np.Inf
        delta_z = torch.norm(z-zk) / np.sqrt(zdim) if k_inner!=0 else np.Inf
        delta = delta_x + delta_z

        if verbose:
            print('ITER %d -->  ' % k, 'beta = %.4f  |  Delta_x: %.5f  |  Delta_z: %.5f  |  MSE to ground-truth: %.5f' % (beta, delta_x, delta_z, compute_mse(xtarget, xk)))

        # Update iters
        x = xk
        z = zk

        if save_iters:
            print('Guardamos puntos limite x^k_infty Y z^k_infty...')
            xiters_jpmap = np.vstack([xiters_jpmap, xk[None,:].cpu().detach().clone()])
            ziters_jpmap = np.vstack([ziters_jpmap, zk.cpu().detach().clone()])
            beta_k.append(beta.cpu().detach().clone())
            indices.append(ind_k)
            ind_k = []

        if (k_inner>=max_iters_inner) or (delta < tol):

            # Update beta
            if uzawa:
                with torch.no_grad():
                    exp_beta = torch.exp(rho*(torch.norm(x-mu_D).pow(2) - alpha))
                    beta = beta * exp_beta

            k += 1  # Increment iter counter of outer loop
            k_inner = 0  # Reset iter counter of inner loop

            if (k>=max_iters) or (delta < tol/100):
                terminate = 1

    if verbose:
        print('Restoration terminated in %d iterations (out of %d)' % (k, max_iters))

    if save_iters:
        return x, z, xiters_jpmap, ziters_jpmap, beta_k, indices
    else:
        return x, z


def csgm(y, A, vae, lamb=0.1, max_iters=5000, tol=1e-5, xtarget=None, zinit=None, device='cuda', verbose=True):
    # Bora, A., Jalal, A., Price, E., & Dimakis, A. G. (2017, August).
    # Compressed sensing using generative models. In Proceedings of
    # the 34th International Conference on Machine Learning-Volume 70
    # (pp. 537-546). JMLR. org.  https://arxiv.org/abs/1703.03208
    #
    # Computes G(argmin_z ||A*G(z) - y||^2 + lamb*||z||^2)

    if verbose:
        print('Running CSGM algorithm...')

    zdim = vae.latent_dim
    vae = vae.to(device)
    decoder = lambda z: vae.decoder(z)[0]

    ### CSGM's loss function:
    def csgm_loss(z, y, A, decoder, lamb):

        ## Datafit = ||A*G(z) - y||^2
        AdotGz = torch.matmul(A, decoder(z).view(-1))
        Datafit = torch.sum((AdotGz - y).pow(2))

        ## Reg = ||z||^2
        Reg = torch.sum(z.pow(2))

        ## lossF = ||A*Dec(z) - xtilde||^2 + lambda*||z||^2
        return (Datafit + lamb*Reg)

    # Initialization
    if zinit is None:
        zinit = torch.zeros(zdim, device=device)
    zk = zinit.requires_grad_(True)
    convergence = False
    k = 0

    # Optimizer
    lr_adam = 0.01
    optimizer = optim.Adam([zk], lr=lr_adam)

    while not convergence and k<max_iters:

        k += 1
        optimizer.zero_grad()

        loss = csgm_loss(zk, y, A, decoder, lamb)
        loss.backward()
        grad = zk.grad

        if torch.norm(grad)/zdim < tol:
            convergence = True
        else:
            optimizer.step()

    if verbose:
        print('Adam terminated in %d iterations (out of %d)' % (k, max_iters))
        print('norm(grad)/zdim at last iteration: %.4f' % (torch.norm(grad)/zdim))

    xk = decoder(zk)[0]

    if xtarget is not None and verbose:
        print(' --> MSE to ground-truth: %.5f' % compute_mse(xtarget, xk))

    return xk, zk


# Auxiliary functions
def compute_mse(x1, x2):

    N = x1.nelement()  # Number of pixels

    return float(torch.sum((x1-x2).pow(2)) / N)


def J1(x,z,A,y,sigma,beta,vae):
    
    with torch.no_grad():
        # Datafit
        DataFit = (0.5/sigma**2)*torch.sum((torch.matmul(A,x.view(-1))-y).pow(2))
        
        # (beta/2)*||xk - Dec(z)||^2
        Acople = 0.5*beta*torch.sum((x - vae.decoder(z)[0]).pow(2))

        # lossF = (1/2)*||z||^2 + (beta/2)*||xk - Dec(z)||^2
        Prior = 0.5*torch.sum(z.pow(2))
    
    return DataFit + Acople + Prior
