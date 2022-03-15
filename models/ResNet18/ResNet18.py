#!/usr/bin/env python3
import warnings

import torch
import torch.nn as nn

class ResNet18ConvBlock(nn.Module):
    """Residual Convolutional Block (ResNet architecture)"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.b1 = nn.BatchNorm2d(out_channels)
        self.a1 = nn.ReLU()
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(out_channels)
        self.a2 = nn.ReLU()

        self.residual = lambda X: X # default

        if out_channels != in_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        Y = self.a1(self.b1(self.c1(X)))
        Y = self.b2(self.c2(Y))
        X = self.residual(X)
        Y += X
        return self.a2(Y)

class ResNet18Module(nn.Module):
    """A residual module using two residual convolution blocks within it"""
    def __init__(self, in_channels, out_channels, halve_spatially=True):
        super().__init__()

        first_block = ResNet18ConvBlock(in_channels, out_channels)
        if halve_spatially:
            first_block = ResNet18ConvBlock(in_channels, out_channels, stride=2)
        second_block = ResNet18ConvBlock(out_channels, out_channels)
        self._layers = nn.Sequential(first_block, second_block)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return self._layers(X)

class NoActionLayer(nn.Module):
    def __call__(self, X: torch.Tensor):
        return X

class ResNet18(nn.Module):
    """ResNet18 achitecture"""
    def __init__(self, n_outputs, output_activation=None):
        super().__init__()
        self._input = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Binary classification: output activation -> sigmoid
        # Multiclass classification: output activation -> softmax
        # Multilabel classification: output activation -> sigmoid

        self._out_activation = output_activation
        # if output_activation is None:
            # warnings.warn("No output_activation function was provided to the ResNet18 __init__ method. Model will assume classification task with num classes defined by 'n_outputs' argument. To avoid this warning, provide the output activation function explicitly")
            # warnings.warn(f"No output_activation function was provided to the ResNet18 __init__ method. Will use {NoActionLayer} as the activation function for the output layer.")
            # self._out_activation = NoActionLayer()
            

        self._layers = nn.Sequential(
            self._input,
            ResNet18Module(64, 64, halve_spatially=False),
            ResNet18Module(64, 128),
            ResNet18Module(128, 256),
            ResNet18Module(256, 512),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, n_outputs),
        )
        if self._out_activation is not None:
            self._layers.add_module("output_activation", self._out_activation)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        return self._layers(X)

if __name__ == "__main__":
    net = ResNet18(3, output_activation=torch.nn.Sigmoid())
    tensor = torch.rand(1, 1, 128, 5168) 
    out = net.forward(tensor)
    print(out.shape)

    for layer in net._layers:
        tensor = layer(tensor)
        print(f"Layer {layer.__class__.__name__} output shape {tensor.shape}")




            
