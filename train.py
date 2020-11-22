import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

from models.dni import DNInn

INPUT_DIMS = 784
IN_CHANNELS = 1
NUM_CLASSES = 10

def make_args_parser():
    # create an ArgumentParser object
    parser = argparse.ArgumentParser()
    # fill parser with information about program arguments
    parser.add_argument('-d', '--device', nargs='+', type=str,
                        choices=['cuda', 'cpu'],
                        default='cpu',
                        help='define the device to train the model')
    parser.add_argument('-e', '--epochs', nargs='+', type=int,
                        default=20000,
                        help='define the number of epochs')
    parser.add_argument('-b', '--batch_size', nargs='+', type=int,
                        default=32,
                        help='define the batch size')
    parser.add_argument('-c', '--conditioned', nargs='+', type=bool,
                        default=False,
                        help='define if the model will be conditioned or not')
    # return an ArgumentParser object
    return parser.parse_args()

def print_args(args):
    print("Running with the following configuration")
    # get the __dict__ attribute of args using vars() function
    args_map = vars(args)
    for key in args_map:
        print('\t', key, '-->', args_map[key])
    # add one more empty line for better output
    print()

def save_grad(name):
    def hook(grad):
        with torch.no_grad():
            backprop_grads[name] = grad
    return hook

def module_optimizer(layer_index, optimizer, forward, images, labels=None):
    optimizer.zero_grad()
    out, grad = forward(layer_index, images, labels)
    out.backward(grad.detach().data)
    optimizer.step()
    out = out.detach()
    return out

def sg_module_optimizer(images, labels, label_onehot, grad_optimizer, optimizer,
                        forward, classification_loss, synthetic_loss):
    grad_optimizer.zero_grad()
    optimizer.zero_grad()
    outs, grads = forward(images, label_onehot)
    global backprop_grads
    if 'backprop_grads' not in globals():
        backprop_grads = {}
    handles = {}
    keys = []
    for i, (out, grad) in enumerate(zip(outs, grads)):
        handles[str(i)] = out.register_hook(save_grad(str(i)))
        keys.append(str(i))
    outputs = outs[-1]
    loss = classification_loss(outputs, labels)
    loss.backward(retain_graph=True)
    for (k, v) in handles.items():
        v.remove()
    grad_loss = 0.
    for k in keys:
        grad_loss += synthetic_loss(grads[int(k)], backprop_grads[k].detach())

    grad_loss.backward()
    grad_optimizer.step()
    return loss, grad_loss

if __name__ == "__main__":
    # parse command line arguments
    args = make_args_parser()
    print_args(args)
    # import datasets
    train_dataset = datasets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = datasets.MNIST(root='./data',
                                train=False,
                                transform=transforms.ToTensor())

    # data Loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False)
    # Define loss functions
    classification_loss = nn.CrossEntropyLoss()
    synthetic_loss = nn.MSELoss()
    # Build model
    net = DNInn().to(args.device)
    # Do training
    print("Start training on {}".format(args.device))
    for epoch in range(args.epochs):
        current = 0
        for i, (images, labels) in enumerate(train_loader):
            current += images.size(0)
            # Convert torch tensor to Variable
            images = Variable(images).to(args.device)
            labels = Variable(labels).to(args.device)
            labels_onehot = torch.zeros([labels.size(0), NUM_CLASSES])
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = Variable(labels_onehot).to(args.device)
            # Train classification model
            out = images
            for index, optimizer in enumerate(net.optimizers):
                if args.conditioned:
                    out = module_optimizer(index, optimizer, net.layer_forward,
                                           out, labels_onehot)
                else:
                    out = module_optimizer(index, optimizer, net.layer_forward, out)
            # Train syntetic gradient models
            loss, grad_loss = sg_module_optimizer(images, labels, labels_onehot,
                                                  net.grad_optimizer, net.optimizer, net,
                                                  classification_loss, synthetic_loss)
            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Grad Loss: %.4f'
                    %(epoch+1, args.epochs, i+1, current//len(train_loader),
                      loss, grad_loss))
