#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import n_atom_features, n_bond_features


class GAT(nn.Module):
    def __init__(
        self,
        n_layers=2,
        n_hidden=50,
        n_heads=1,
        dropout=0,
        n_atom_features=n_atom_features,
        n_bond_features=n_bond_features,
        regression=True,
        n_tasks=1,
    ):
        super().__init__()
        self.n_layers = int(n_layers)
        assert n_layers > 0
        self.n_atom_features = int(n_atom_features)
        self.n_bond_features = int(n_bond_features)
        self.n_hidden = int(n_hidden)
        self.n_tasks = int(n_tasks)
        self.dropout = dropout
        self.regression = regression
        self.layers = nn.ModuleList()
        self.n_heads = int(n_heads)

        self.layers.append(
            MultiHeadGATLayer(
                self.n_atom_features,
                self.n_hidden,
                0,
                self.n_heads,
                activation=F.selu_,
            )
        )
        for _ in range(1, self.n_layers):
            self.layers.append(
                MultiHeadGATLayer(
                    self.n_hidden * self.n_heads,
                    self.n_hidden,
                    self.dropout,
                    self.n_heads,
                    activation=F.selu_,
                )
            )
        self.layers.append(
            MultiHeadGATLayer(
                self.n_hidden * self.n_heads,
                self.n_hidden,
                self.dropout,
                self.n_heads,
                activation=None,
            )
        )

        self.readout = MeanReadout()
        self.mlp = nn.Sequential(
            nn.Linear(self.n_hidden * self.n_heads, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_tasks),
        )

    def forward(
        self, node_features, edge_features, edge_domain, edge_range, graph_id
    ):
        h = node_features
        for layer in self.layers:
            h = layer(h, edge_domain, edge_range)
        graph_embedding = self.readout(h, graph_id)
        out = self.mlp(graph_embedding)
        return out


class MeanReadout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, nodes, graph_id):
        batchsize = int(torch.max(graph_id) + 1)
        graph_sum = nodes.new_zeros((batchsize, nodes.shape[1]))
        graph_dom = nodes.new_zeros((batchsize), dtype=torch.int)
        ones = nodes.new_ones((graph_id.shape[0]), dtype=torch.int)
        graph_sum.index_add_(0, graph_id, nodes)
        graph_dom.index_add_(0, graph_id, ones)
        graph_embedding = graph_sum / graph_dom.view(-1, 1)
        return graph_embedding


class MultiHeadGATLayer(nn.Module):
    def __init__(
        self,
        n_features,
        n_output,
        dropout,
        n_heads,
        activation=F.relu_,
        merge_func="cat",
    ):
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(n_heads):
            self.heads.append(
                GATLayer(n_features, n_output, dropout, activation=activation)
            )
        self.merge = merge_func

    def forward(self, node_features, edge_domain, edge_range):
        head_outs = [
            head(node_features, edge_domain, edge_range) for head in self.heads
        ]

        if self.merge == "cat":
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GATLayer(nn.Module):
    def __init__(
        self,
        n_features,
        n_output,
        dropout=0,
        attn_dropout=0,
        activation=F.relu_,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_output = n_output
        self.dropout_prob = dropout
        self.attn_dropout_prob = attn_dropout

        self.dropout = nn.Dropout(p=self.dropout_prob)
        self.attn_dropout = nn.Dropout(p=self.attn_dropout_prob)

        self.W = nn.Linear(n_features, n_output, bias=False)
        self.a = nn.Linear(n_output * 2, 1, bias=False)

        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.W.weight, gain=gain)
        nn.init.xavier_uniform_(self.a.weight, gain=gain)

    def forward(self, node_features, edge_domain, edge_range):
        """
            Args:
                node_features: Tensor(n_nodes, n_features)
                edge_domain: Tensor(n_edges)
                edge_range: Tensor(n_edges)
            Returns:
                out: Tensor(n_nodes, n_output)
        """
        z = self.W(self.dropout(node_features))  # (n_nodes, )
        n_nodes = node_features.shape[0]
        # edge attention
        edge_self = edge_range.new_tensor(np.arange(n_nodes, dtype=np.long))
        edge_domain2 = torch.cat((edge_domain, edge_self), 0)
        edge_range2 = torch.cat((edge_range, edge_self), 0)
        edge_range_sorted, idx = torch.sort(edge_range2, 0)
        edge_domain_sorted = torch.gather(edge_domain2, 0, idx)

        domain_z = torch.index_select(z, 0, edge_domain_sorted)
        range_z = torch.index_select(z, 0, edge_range_sorted)
        cat_z = torch.cat((domain_z, range_z), 1)
        e_ij = F.leaky_relu(self.a(cat_z))  # (n_edges, 1)

        # message
        partitioned = dynamic_partition(e_ij, edge_range_sorted, int(n_nodes))
        alpha = vec_softmax(
            partitioned
        )  # torch.cat([F.softmax(es, 0) for es in partitioned], 0)
        alpha = self.attn_dropout(alpha)
        # (edges, 1)

        out = node_features.new_zeros((n_nodes, self.n_output))
        alpha_z = alpha * domain_z
        out.index_add_(0, edge_range_sorted, alpha_z)
        if self.activation is not None:
            out = self.activation(out)
        return out  # (n_nodes, n_output)


@torch.jit.script
def vec_softmax(partitioned: List[torch.Tensor]):
    return torch.cat([F.softmax(es, 0) for es in partitioned], 0)


@torch.jit.script
def dynamic_partition(
    data: torch.Tensor, partitions: torch.Tensor, num_partitions: int
):
    res = []
    for i in range(num_partitions):
        b = partitions == i
        res += [data[torch.nonzero(b).squeeze(1)]]
    return res
