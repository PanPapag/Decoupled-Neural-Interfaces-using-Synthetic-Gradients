import torch
import torch.nn as nn

from collections import OrderedDict

from .sg import SyntheticGradients

class DNInn(nn.Module):
    def __init__(self, input_size=784, hidden_layer_sizes=[1024], num_classes=10, conditioned=False):
        super(DNInn, self).__init__()
        self.no_core_layers = len(hidden_layer_sizes) + 1
        # Core network
        self.core_layers = OrderedDict()
        for i, _ in enumerate(hidden_layer_sizes):
            layer_name = 'layer_' + str(i+1)
            if i == 0:
                self.core_layers[layer_name] = nn.Linear(input_size, hidden_layer_sizes[i])
                self.core_layers['relu_' + str(i+1)] = nn.ReLU()
            else:
                self.core_layers[layer_name] = nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i])
                self.core_layers['relu_' + str(i+1)] = nn.ReLU()
            if i == len(hidden_layer_sizes) - 1:
                layer_name = 'layer_' + str(i+2)
                self.core_layers[layer_name] = nn.Linear(hidden_layer_sizes[i], num_classes)

        # Syntehtic gradients
        self.sg_layers = OrderedDict()
        index = 0
        for i, layer_name in enumerate(self.core_layers.keys()):
            if i == len(self.core_layers.keys()) - 1:
                self.sg_layers['SG_' + layer_name] = \
                                        SyntheticGradients(
                                            input_size=num_classes,
                                            num_classes=num_classes,
                                            conditioned=conditioned
                                        )
            else:
                if 'layer' in layer_name:
                    self.sg_layers['SG_' + layer_name] = \
                                        SyntheticGradients(
                                            input_size=hidden_layer_sizes[index],
                                            num_classes=num_classes,
                                            conditioned=conditioned
                                        )
                    index += 1

        # Final Model
        self.dni_nn = nn.ModuleDict(self.core_layers)
        self.sg_nn = nn.ModuleDict(self.sg_layers)

        self.optimizers = []
        self.init_optimzers()

    def init_optimzers(self, learning_rate=3e-5):
        for layer_name, layer in self.core_layers.items():
            if 'layer' in layer_name:
                self.optimizers.append(torch.optim.Adam(layer.parameters(), lr=learning_rate))
        self.optimizer = torch.optim.Adam(self.dni_nn.parameters(), lr=learning_rate)
        self.grad_optimizer = torch.optim.Adam(self.sg_nn.parameters(), lr=learning_rate)

    def layer_forward(self, index, x, y=None):
        core_layer_name = 'layer_' + str(index+1)
        sg_layer_name = 'SG_' + core_layer_name
        # print(core_layer_name, sg_layer_name, x.size())
        if index == 0:
            x = x.view(-1, 28*28)
        elif index+1 == self.no_core_layers:
            x = x.view(x.size(0), -1)
        else:
            out = self.dni_nn['relu_' + str(index+1)](x)
        out = self.dni_nn[core_layer_name](x)
        grad = self.sg_nn[sg_layer_name](out, y)
        return out, grad

    def forward(self, x, y=None):
        x = x.view(-1, 28*28)
        outs = []
        fc_outs = []
        for i, layer in enumerate(self.dni_nn.values()):
            if i == 0:
                outs.append(layer(x))
            else:
                outs.append(layer(outs[-1]))
            if isinstance(layer, nn.Linear):
                fc_outs.append(outs[-1])

        if y is not None:
            grads_fc = []
            for i, fc_out in enumerate(fc_outs):
                layer_name = 'SG_layer_' + str(i+1)
                grads_fc.append(self.sg_nn[layer_name](fc_out, y))
            return (*fc_outs,), (*grads_fc,)
        else:
            return (*fc_outs,)

# Test code
if __name__ == "__main__":
    m = DNInn(hidden_layer_sizes=[1024])
    print(m)
