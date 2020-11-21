import torch
import torch.nn as nn
import numpy as np

class SyntheticGradients(nn.Module):
    def __init__(self, input_size=784, hidden_layer_sizes=[1024, 1024], num_classes=10, conditioned=False):
        super(SyntheticGradients, self).__init__()
        self.input_size = input_size
        self.conditioned = conditioned
        if self.conditioned:
            sg_input_size = self.input_size + num_classes
        else:
            sg_input_size = self.input_size

        layers = []
        for i, _ in enumerate(hidden_layer_sizes):
            if i == 0:
                layers.append(nn.Sequential(
                                nn.Linear(sg_input_size, hidden_layer_sizes[i]),
                                nn.BatchNorm1d(hidden_layer_sizes[i]),
                                nn.ReLU(inplace=True))
                             )
            else:
                layers.append(nn.Sequential(
                                nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]),
                                nn.BatchNorm1d(hidden_layer_sizes[i]),
                                nn.ReLU(inplace=True))
                             )
            if i == len(hidden_layer_sizes) - 1:
                layers.append(nn.Linear(hidden_layer_sizes[i], self.input_size))

            self.layers = nn.ModuleList(layers)

    def forward(self, x, y):
        if self.conditioned:
            assert y is not None
            x = torch.cat((x, y), 1)
        outs = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                outs.append(layer(x))
            else:
                outs.append(layer(outs[-1]))
        return outs[-1]

# Test code
if __name__ == "__main__":
    sg = SyntheticGradients(hidden_layer_sizes=[1024, 2048, 1024], conditioned=True)
    print(sg)
