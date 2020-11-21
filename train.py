import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.autograd import Variable

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
                        default=1,
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
