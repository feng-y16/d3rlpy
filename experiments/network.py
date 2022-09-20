import torch
import torch.nn as nn
import numpy as np
import pdb
from copy import deepcopy
import itertools
from torch.optim import Adam
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
import random


def mlp(sizes, activation=nn.ELU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
        nn.init.orthogonal_(layers[-2].weight)
    return nn.Sequential(*layers)


class MLPAutoencoder(nn.Module):
    """A salt-of-the-earth MLP Autoencoder + some edgy res connections"""

    def __init__(self, input_size, hidden_size, latent_size):
        super(MLPAutoencoder, self).__init__()
        self.encoder = mlp([input_size] + [hidden_size] * 3 + [latent_size])
        self.decoder = mlp([latent_size] + [hidden_size] * 3 + [input_size])

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat


class ODEFunc(nn.Module):
    def __init__(self, latent_size, action_size, hidden_size):
        super().__init__()
        self.latent_size = latent_size
        self.h_mlp = mlp([latent_size] + [hidden_size] * 3 + [1])
        self.a_mlp = mlp([latent_size + action_size] + [hidden_size] * 3 + [latent_size // 2])
        self.a = None

    def forward(self, t, x):
        with torch.enable_grad():
            xx = x.detach()
            xx.requires_grad = True
            grad = torch.autograd.grad(self.h_mlp(xx).sum(), xx, create_graph=True, retain_graph=True)[0]
            grad_q = grad[:, self.latent_size // 2:].detach()
            grad_p = grad[:, : self.latent_size // 2].detach()
            grad[:, self.latent_size // 2:] = grad_p
            grad[:, :self.latent_size // 2] = -grad_q + self.a_mlp(torch.cat((x, self.a), dim=1))
        return grad


class EventFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = mlp([input_dim] + [hidden_dim] * 3 + [output_dim])

    def forward(self, x):
        return self.mlp(x)


class Baseline(nn.Module):
    def __init__(self, observation_shape, action_shape, hidden_size_ode=256, hidden_size_ae=256, latent_size=32,
                 **kwargs):
        super(Baseline, self).__init__()
        if type(observation_shape) is tuple:
            observation_size = observation_shape[0]
        else:
            observation_size = observation_shape
        if type(action_shape) is tuple:
            action_size = action_shape[0]
        else:
            action_size = action_shape
        self.observation_size = observation_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.pred_nn = mlp([observation_size + action_size] + [hidden_size_ode] * 5 + [observation_size])
        self.rew_nn = mlp([observation_size + action_size] + [hidden_size_ae] * 5 + [2])
        self.o = None

    def set_o(self, o):
        if len(o.size()) == 1:
            self.o = o.unsqueeze(0)
        else:
            self.o = o

    def forward(self, o, a):
        o2 = self.pred_nn(torch.cat((o, a), dim=1))
        o_recon = o
        rd = self.rew_nn(torch.cat((o, a), dim=1))
        r = rd[:, 0]
        d = 1 / (1 + torch.exp(-rd[:, 1].clamp(-10, 10)))
        return o2, o_recon, r, d

    def step(self, a):
        if len(a.size()) == 1:
            a = a.unsqueeze(0)
        with torch.no_grad():
            o2 = self.pred_nn(torch.cat((self.o, a), dim=1))
            rd = self.rew_nn(torch.cat((self.o, a), dim=1))
            r = rd[:, 0]
            d = 1 / (1 + torch.exp(-rd[:, 1].clamp(-10, 10)))
        self.o = o2
        return self.o, r, d


class NODA(nn.Module):

    def __init__(self, dataset, observation_space, action_space, hidden_size_ode=256, hidden_size_ae=256,
                 latent_size=32, use_ode=True, tol=1e-6, device='cpu'):
        super(NODA, self).__init__()
        self.dataset = dataset
        self.observation_space = observation_space
        self.action_space = action_space
        if type(observation_space.shape) is tuple:
            observation_size = observation_space.shape[0]
        else:
            observation_size = observation_space.shape
        if type(action_space.shape) is tuple:
            action_size = action_space.shape[0]
        else:
            action_size = action_space.shape
        self.observation_size = observation_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.ae = MLPAutoencoder(observation_size, hidden_size_ae, latent_size)
        self.integration_time = torch.as_tensor([0, 1], dtype=torch.float32, device=device)
        self.ode_func = ODEFunc(latent_size, action_size, hidden_size_ode)
        self.ae_mlp = mlp([latent_size + action_size] + [hidden_size_ode] * 3 + [latent_size])
        # self.event_func = EventFunc(self.latent_dim + self.action_shape, hidden_dim_ode, 1)
        # self.post_process_func = mlp([self.latent_dim] + [hidden_dim_ode] * 3 + [self.latent_dim])
        self.rew_nn = mlp([2 * latent_size + action_size] + [hidden_size_ae] * 3 + [2])
        self.use_ode = use_ode
        self.tol = tol
        self.device = device
        self.o = None
        self.latent_s = None
        self.optimizer = None

    def reset(self, seed=0):
        np.random.seed(seed)
        o = self.dataset.observations[np.random.randint(0, len(self.dataset.observations))]
        self.set_o(torch.as_tensor(o, dtype=torch.float32, device=self.device))
        return o

    def set_o(self, o):
        if len(o.size()) == 1:
            self.o = o.unsqueeze(0)
        else:
            self.o = o
        self.latent_s = self.ae.encode(self.o)

    def forward(self, o, a):
        latent_s = self.ae.encode(o)
        if self.use_ode:
            self.ode_func.a = a
            # mask = self.event_func.forward(torch.cat((latent_s, a), dim=1)) > 0
            # latent_s2 = latent_s + ~mask * self.post_process_func(latent_s)
            latent_s2 = latent_s
            latent_s2 = odeint(self.ode_func, latent_s2, self.integration_time,
                               rtol=self.tol, atol=self.tol)[1]
        else:
            # mask = self.event_func.forward(torch.cat((latent_s, a), dim=1)) > 0
            # latent_s2 = latent_s + ~mask * self.post_process_func(latent_s)
            latent_s2 = latent_s
            latent_s2 = self.ae_mlp(torch.cat((latent_s2, a), dim=1))
        o2 = self.ae.decode(latent_s2)
        o_recon = self.ae.decode(latent_s)
        rd = self.rew_nn(torch.cat((latent_s, latent_s2, a), dim=1))
        r = rd[:, 0]
        d = 1 / (1 + torch.exp(-rd[:, 1].clamp(-10, 10)))
        return o2, o_recon, r, d

    def step(self, a):
        a = torch.as_tensor(a, dtype=torch.float32, device=self.device)
        if len(a.shape) == 1:
            a = a.unsqueeze(0)
        if self.use_ode:
            self.ode_func.a = a
            # mask = self.event_func.forward(torch.cat((self.latent_s, a), dim=1)) > 0
            # latent_s2 = self.latent_s + ~mask * self.post_process_func(self.latent_s)
            latent_s2 = self.latent_s
            latent_s2 = odeint(self.ode_func, latent_s2, self.integration_time,
                               rtol=self.tol, atol=self.tol)[1]
        else:
            with torch.no_grad():
                # mask = self.event_func.forward(torch.cat((self.latent_s, a), dim=1)) > 0
                # latent_s2 = self.latent_s + ~mask * self.post_process_func(self.latent_s)
                latent_s2 = self.latent_s
                latent_s2 = self.ae_mlp(torch.cat((latent_s2, a), dim=1))
        with torch.no_grad():
            self.o = self.ae.decode(latent_s2)
            rd = self.rew_nn(torch.cat((self.latent_s, latent_s2, a), dim=1))
            r = rd[:, 0]
            d = torch.where(rd[:, 1] >= 0.5, 1, 0).int()
        self.latent_s = latent_s2
        return self.o.cpu().numpy()[0], r.cpu().numpy()[0], d.cpu().numpy()[0], {}

    def forward_model(self, data, train=True, loss_weights=None):
        if loss_weights is None:
            loss_weights = [1, 1, 1, 1]
        o, a, r, o2, d = data
        if not train:
            with torch.no_grad():
                o2_pred, o_recon, r_pred, d_pred = self.forward(o, a)
        else:
            o2_pred, o_recon, r_pred, d_pred = self.forward(o, a)
        loss_o_pred = ((o2_pred - o2) ** 2).mean()
        loss_o_recon = ((o_recon - o) ** 2).mean()
        loss_r_pred = ((r_pred - r) ** 2).mean()
        loss_d_pred = (-d * torch.log(d_pred) - (1 - d) * torch.log(1 - d_pred)).mean()

        loss_model = loss_weights[0] * loss_o_pred + loss_weights[1] * loss_o_recon + \
                     loss_weights[2] * loss_r_pred + loss_weights[3] * loss_d_pred
        if train:
            model_info = dict(LMOPred=loss_o_pred.item(),
                              LMORecon=loss_o_recon.item(),
                              LMRPred=loss_r_pred.item(),
                              LMDPred=loss_d_pred.item(),
                              LMT=loss_model.item())
        else:
            model_info = dict(TLMOPred=loss_o_pred.item(),
                              TLMORecon=loss_o_recon.item(),
                              TLMRPred=loss_r_pred.item(),
                              TLMDPred=loss_d_pred.item(),
                              TLMT=loss_model.item())
        if train:
            self.optimizer.zero_grad()
            loss_model.backward()
            self.optimizer.step()
        return model_info

    def fit(self, train_dataloader, epochs, lr=1e-3):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_batch_num = len(train_dataloader)
        for epoch in range(1, epochs + 1):
            pbar = tqdm(total=train_batch_num, desc='Epoch {:}'.format(epoch))
            for i, train_data in enumerate(train_dataloader):
                self.train()
                info = self.forward_model(train_data)
                pbar.set_postfix(**info)
                # logger.store(**info)
                pbar.update()
            pbar.close()
