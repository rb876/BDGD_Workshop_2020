import odl
import abc
import numpy as np
import torch
from odl.contrib import torch as odl_torch

class ForwardModel:
    def __init__(self):
        self.space = None
        self.geometry = None
        self.operator = None
        self.adjoint = None
        self.pseudoinverse = None

    @property
    def space(self):
        return self.__space
    @space.setter
    def space(self, space):
        self.__space = space

    @property
    def geometry(self):
        return self.__geometry
    @geometry.setter
    def geometry(self, geometry):
        self.__geometry = geometry

    @property
    def operator(self):
        return self.__operator
    @operator.setter
    def operator(self, operator):
        self.__operator = operator

    @property
    def adjoint(self):
        return self.__adjoint
    @adjoint.setter
    def adjoint(self, adjoint):
        self.__adjoint = adjoint

    @property
    def pseudoinverse(self):
        return self.__pseudoinverse
    @pseudoinverse.setter
    def pseudoinverse(self, pseudoinverse):
        self.__pseudoinverse = pseudoinverse

class SimpleCT(ForwardModel):
    def __init__(self):
        super().__init__()

    def sinogram(self, phantom):
        clean = self.operator(phantom)
        scale = torch.abs(clean).view(clean.shape[0], -1).mean(1, keepdim=True)
        noise = clean.data.new( clean.size()).normal_(0, 1) * \
            scale.view(-1, 1, 1).repeat(1, clean.shape[-2], clean.shape[-1]) * 0.01
        noisy = clean + noise
        fbp = self.pseudoinverse(noisy)
        return noisy, phantom, fbp

    def grad(self, x, y):
        return self.adjoint(self.operator(x) - y)
