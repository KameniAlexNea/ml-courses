import os
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class GCNConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        # Initialize the parameters.
        stdv = 1. / math.sqrt(out_channels)
        self.theta.data.uniform_(-stdv, stdv)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        TODO:
            1. Generate the adjacency matrix with self-loop \hat{A} using edge_index.
            2. Calculate the diagonal degree matrix \hat{D}.
            3. Calculate the output X' with torch.mm using the equation above.
        """
        # your code here
        D_out, _ = x.shape
        A = torch.zeros(D_out, D_out, device=x.device)
        for (e1, e2) in edge_index.T:
            A[e1, e2] = 1
        A_p = A + torch.eye(D_out, device=A.device)
        D_p = torch.zeros_like(A)
        for i, a in enumerate(torch.sum(A_p, dim=-1)):
            D_p[i, i] = 1/torch.sqrt(a)
        ret_a = torch.mm(torch.mm(D_p, A_p), D_p)
        ret_b = torch.mm(x, self.theta)
        ret = torch.mm(ret_a, ret_b)
        return ret

# from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        """
        TODO:
            1. Define the first convolution layer using `GCNConv()`. Set `out_channels` to 64;
            2. Define the first activation layer using `nn.ReLU()`;
            3. Define the second convolution layer using `GCNConv()`. Set `out_channels` to 64;
            4. Define the second activation layer using `nn.ReLU()`;
            5. Define the third convolution layer using `GCNConv()`. Set `out_channels` to 64;
            6. Define the dropout layer using `nn.Dropout()`;
            7. Define the linear layer using `nn.Linear()`. Set `output_size` to 2.

        Note that for MUTAG dataset, the number of node features is 7, and the number of classes is 2.

        """
        
        # your code here
        self.first = GCNConv(7, 64)
        self.first_act = nn.ReLU()
        
        self.second = GCNConv(64, 64)
        self.second_act = nn.ReLU()
        
        self.third = GCNConv(64, 64)
        self.third_act = nn.ReLU()
        
        self.linear = nn.Linear(64, 2)

    def forward(self, x, edge_index, batch):
        """
        TODO:
            1. Pass the data through the frst convolution layer;
            2. Pass the data through the activation layer;
            3. Pass the data through the second convolution layer;
            4. Obtain the graph embeddings using the readout layer with `global_mean_pool()`;
            5. Pass the graph embeddgins through the dropout layer;
            6. Pass the graph embeddings through the linear layer.
            
        Arguments:
            x: [num_nodes, 7], node features
            edge_index: [2, num_edges], edges
            batch: [num_nodes], batch assignment vector which maps each node to its 
                   respective graph in the batch

        Outputs:
            probs: probabilities of shape (batch_size, 2)
        """
        
        # your code here
        x = self.first_act(self.first(x, edge_index))
        x = self.second_act(self.second(x, edge_index))
        x = self.third(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.third_act(x)
        return self.linear(x)