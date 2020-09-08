import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mean_field_VI.utils as utils
from neural_trainer import DeterministicTrainer, BayesianTrainer
from layers import LocalReparamConv2d


class DeterministicBlock(DeterministicTrainer):
    def __init__(self, arch_args):
        super().__init__()
        self.upper_layers, self.lower_layers, self.common_layers = \
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList()

        for shape in arch_args['mean']['upper']:
            layer = torch.nn.Conv2d
            self.upper_layers.append(layer( * shape))

        for shape in arch_args['mean']['lower']:
            layer = torch.nn.Conv2d
            self.lower_layers.append(layer( * shape))

        for shape in arch_args['mean']['common']:
            layer = torch.nn.Conv2d
            self.common_layers.append(layer( * shape))

    def forward(self, x, grad):
        skip = x
        for layer in self.upper_layers:
            x = layer(x)
            x = F.relu(x)
        for layer in self.lower_layers:
            grad = layer(grad)
            grad = F.relu(grad)
        x = torch.cat((x, grad), dim=1)
        for idx, layer in enumerate(self.common_layers):
            x = layer(x)
            if idx < len(self.common_layers)-1:
                x = F.relu(x)
            else:
                x = x
        return F.relu(x + skip), utils.to_gpu(torch.ones(1))

    def compute_loss(self, x, x_pred, batch_size, data_size, var):
        log_likelihood = self.compute_log_likelihood(x, x_pred, var)
        return -log_likelihood

    def compute_log_likelihood(self, x, x_pred, var):
        return torch.sum(utils.gaussian_log_density(inputs=utils._squeeze(x), mean=utils._squeeze(x_pred), variance=var), dim=0)

class BlockHomo(BayesianTrainer):
    def __init__(self, arch_args):
        super().__init__()
        self.det_upper_layers, self.det_lower_layers, self.det_common_layers =\
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.log_std = nn.Parameter(torch.zeros(1))

        for shape in arch_args['mean']['upper']:
            layer = torch.nn.Conv2d
            self.det_upper_layers.append(layer( * shape))

        for shape in arch_args['mean']['lower']:
            layer = torch.nn.Conv2d
            self.det_lower_layers.append(layer( * shape))

        for shape in arch_args['mean']['common'][:-1]:
            layer = torch.nn.Conv2d
            self.det_common_layers.append(layer( * shape))

        self.bayes_CNN = LocalReparamConv2d(arch_args['mean']['common'][-1])

    def forward(self, x, grad, num_samples=1):
        skip = x
        x = self.encode(x, grad)
        x = self.bayes_CNN(x, num_samples=num_samples, squeeze=True)
        return F.relu(x + skip), self.log_std.exp()**2

    def encode(self, x, grad):
        for layer in self.det_upper_layers:
            x = layer(x)
            x = F.relu(x)
        for layer in self.det_lower_layers:
            grad = layer(grad)
            grad = F.relu(grad)
        x = torch.cat((x, grad), dim=1)
        for layer in self.det_common_layers:
            x = layer(x)
            x = F.relu(x)
        return x

    def compute_loss(self, x, x_pred, batch_size, data_size, var):
        # The objective is 1/n * (\sum_i log_like_i - KL)
        log_likelihood = self.compute_log_likelihood(x, x_pred, var)
        kl = self.compute_kl() * (batch_size / data_size)
        elbo = log_likelihood - kl
        return -elbo, kl

    def compute_kl(self):
        return self.bayes_CNN.compute_kl()

    def compute_log_likelihood(self, x, x_pred, var):
        return torch.sum(utils.gaussian_log_density(inputs=utils._squeeze(x), mean=utils._squeeze(x_pred), variance=var), dim=0)


class BlockHetero(BayesianTrainer):
    def __init__(self, arch_args):
        super().__init__()
        self.det_upper_CNN_mean, self.det_lower_CNN_mean, self.det_common_CNN_mean, self.det_CNN_log_std = \
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for shape in arch_args['mean']['upper']:
            layer = torch.nn.Conv2d
            self.det_upper_CNN_mean.append(layer( * shape))

        for shape in arch_args['mean']['lower']:
            layer = torch.nn.Conv2d
            self.det_lower_CNN_mean.append(layer( * shape))

        for shape in arch_args['mean']['common'][: -1]:
            layer = torch.nn.Conv2d
            self.det_common_CNN_mean.append(layer( * shape))

        for shape in arch_args['variance'][: -1]:
            layer = torch.nn.Conv2d
            self.det_CNN_log_std.append(layer( * shape))

        self.bayes_CNN_mean = LocalReparamConv2d(arch_args['mean']['common'][-1])
        self.bayes_CNN_log_std = LocalReparamConv2d(arch_args['variance'][-1])

    def forward(self, x, grad, num_samples=1):
        mean = self.mean(x, grad, num_samples=num_samples)
        var = self.variance(x, grad, num_samples=num_samples)
        return mean, var

    def encode(self, x, grad):
        for layer in self.det_upper_CNN_mean:
            x = layer(x)
            x = F.relu(x)
        for layer in self.det_lower_CNN_mean:
            grad = layer(grad)
            grad = F.relu(grad)
        x = torch.cat((x, grad), dim=1)
        for layer in self.det_common_CNN_mean:
            x = layer(x)
            x = F.relu(x)
        return x

    def mean(self, x, grad, num_samples):
        skip = x
        x = self.encode(x, grad)
        x = self.bayes_CNN_mean(x, num_samples=num_samples, squeeze=True)
        return F.relu(skip + x)

    def variance(self, x, grad, num_samples):
        log_std = torch.cat((x, grad), dim=1)
        for layer in self.det_CNN_log_std:
            log_std = layer(log_std)
            log_std = F.relu(log_std)
        return F.softplus(self.bayes_CNN_log_std(log_std, num_samples=num_samples, squeeze=True))**2 + 1e-6

    def compute_loss(self, x, x_pred, batch_size, data_size, var):
        # The objective is 1/n * (\sum_i log_like_i - KL)
        log_likelihood = self.compute_log_likelihood(x, x_pred, var)
        kl = self.compute_kl() * (batch_size / data_size)
        elbo = log_likelihood - kl
        return -elbo, kl

    def compute_kl(self):
        return self.bayes_CNN_mean.compute_kl() + self.bayes_CNN_log_std.compute_kl()

    def compute_log_likelihood(self, x, x_pred, var):
        return torch.sum(utils.gaussian_log_density(inputs=utils._squeeze(x), mean=utils._squeeze(x_pred), variance=utils._squeeze(var)), dim=0)
