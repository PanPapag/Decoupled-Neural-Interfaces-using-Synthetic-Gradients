# Decoupled-Neural-Interfaces-using-Synthetic-Gradients

A PyTorch implementation of the paper [Decoupled Neural Interfaces using Synthetic Gradients](https://arxiv.org/abs/1608.05343). 
This repo is a modification of [dni.pytorch](https://github.com/andrewliao11/dni.pytorch0). In this implementation both classification and 
synethic gradient models are Feed-Forward NN which can be modified rather simple just by changing the hidden_layer_sizes parameters. 

## Introduction 

Training directed neural networks typically requires forward-propagating data through a computation graph, followed by backpropagating error signal, to produce weight updates. All layers, or more generally, modules, of the network are therefore locked, in the sense that they must wait for the remainder of the network to execute forwards and propagate error backwards before they can be updated. In this work we break this constraint by decoupling modules by introducing a model of the future computation of the network graph. These models predict what the result of the modelled subgraph will produce using only local information. In particular we focus on modelling error gradients: by using the modelled synthetic gradient in place of true backpropa- gated error gradients we decouple subgraphs, and can update them independently and asynchronously i.e. we realise decoupled neural interfaces. 

![https://github.com/PanPapag/Decoupled-Neural-Interfaces-using-Synthetic-Gradients/blob/main/misc/dni.png]()
