import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mean_field_VI.utils as utils

class DeterministicTrainer(nn.Module):
    def __init__(self):
        super().__init__()

    def optimise(self, train_loader, epochs=100, batch_size=64, initial_lr=1e-3, weight_decay=1e-3):

        weights = [v for k, v in self.named_parameters()]
        optimizer = optim.Adam([{'params': weights, 'weight_decay': weight_decay}], lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4)
        for epoch in range(epochs):
            scheduler.step()
            losses, rmses, psnrs = [], [], []
            for (data, grad, target) in train_loader:
                self.train()
                optimizer.zero_grad()
                data, grad, target = utils.to_gpu(data, grad, target)
                x_pred, var = self.forward(data, grad)
                step_loss = self.compute_loss(target.squeeze(), x_pred.squeeze(), len(target), len(train_loader.dataset), var)
                step_loss.backward()
                optimizer.step()
                # evaluation
                rmse, psnr = self._evaluate_performance(target, x_pred)
                losses.append(step_loss.cpu().item())
                rmses.append(rmse)
                psnrs.append(psnr)

            print('epoch: {}, loss: {:.4f}, rmse: {:.8f}, psnr: {:.8f}'\
                .format(epoch, np.mean(losses), np.mean(rmses), np.mean(psnrs)), flush=True)

    def _evaluate_performance(self, x, x_pred):
        from math import log10
        return (torch.sqrt(torch.mean((x - x_pred) ** 2)).cpu().item(), 10
                * log10(1 / torch.mean((x - x_pred) ** 2)))


class BayesianTrainer(nn.Module):
    def __init__(self):
        super().__init__()

    def optimise(self, train_loader, epochs=100, batch_size=64, initial_lr=1e-3, weight_decay=1e-3):

        deterministic_weights  = [v for k, v in self.named_parameters() if k.startswith('det_')]
        variational_weights = [v for k, v in self.named_parameters() if (not k.startswith('det_'))]
        optimizer = optim.Adam([{'params': deterministic_weights, 'weight_decay': weight_decay},
                                {'params': variational_weights}], lr=initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4)
        for epoch in range(epochs):
            scheduler.step()
            losses, kls, rmses, psnrs = [], [], [], []
            for (data, grad, target) in train_loader:
                self.train()
                optimizer.zero_grad()
                data, grad, target = utils.to_gpu(data, grad, target)
                x_pred, var = self.forward(data, grad, num_samples=1)
                step_loss, kl = self.compute_loss(target.squeeze(), x_pred.squeeze(), len(target), len(train_loader.dataset), var)
                step_loss.backward()
                optimizer.step()
                # evaluation
                rmse, psnr = self._evaluate_performance(target, x_pred)
                losses.append(step_loss.cpu().item())
                kls.append(kl.cpu().item())
                rmses.append(rmse)
                psnrs.append(psnr)

            print('epoch: {}, loss: {:.4f}, kl: {:.4f}, rmse: {:.8f}, psnr: {:.8f}'\
                .format(epoch, np.mean(losses), np.mean(kls), np.mean(rmses), np.mean(psnrs)), flush=True)

    def _evaluate_performance(self, x, x_pred):
        from math import log10
        return (torch.sqrt(torch.mean((x - x_pred) ** 2)).cpu().item(), 10
                * log10(1 / torch.mean((x - x_pred) ** 2)))
