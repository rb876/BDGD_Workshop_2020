import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import mean_field_VI.utils as utils
from neural_trainer import NeuralTrainer
from layers import LocalReparamConv2d

class BlockHetero(NeuralTrainer):
    def __init__(self, arch_args):
        super().__init__()
        self.det_upper_CNN_mean, self.det_lower_CNN_mean, self.det_common_CNN_mean, self.det_CNN_variance = \
            nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for shape in arch_args['mean']['upper']:
            layer = torch.nn.Conv2d
            self.det_upper_CNN_mean.append(layer(*shape))

        for shape in arch_args['mean']['lower']:
            layer = torch.nn.Conv2d
            self.det_lower_CNN_mean.append(layer(*shape))

        for shape in arch_args['mean']['common'][: -1]:
            layer = torch.nn.Conv2d
            self.det_common_CNN_mean.append(layer(*shape))

        for shape in arch_args['variance'][: -1]:
            layer = torch.nn.Conv2d
            self.det_CNN_variance.append(layer(*shape))

        self.bayes_CNN_mean = LocalReparamConv2d(arch_args['mean']['common'][-1])
        self.bayes_CNN_variance = LocalReparamConv2d(arch_args['variance'][-1])

    def forward(self, x, grad, num_samples=1):
        return self.mean(x, grad, num_samples=num_samples), self.variance(x, grad, num_samples=num_samples)

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
        rho = torch.cat((x, grad), dim=1)
        for layer in self.det_CNN_variance:
            rho = layer(rho)
            rho = F.relu(rho)
        return F.softplus(self.bayes_CNN_variance(rho, num_samples=num_samples, squeeze=True))**2 + 1e-6

    def compute_loss(self, x, x_pred, batch_size, data_size, var):
        # The objective is 1/n * (\sum_i log_like_i - KL)
        log_likelihood = self.compute_log_likelihood(x, x_pred, var)
        kl = self.compute_kl() * (batch_size / data_size)
        elbo = log_likelihood - kl
        return -elbo, kl

    def compute_kl(self):
        return self.bayes_CNN_mean.compute_kl() + self.bayes_CNN_variance.compute_kl()
