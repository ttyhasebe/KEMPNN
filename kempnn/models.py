#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import n_atom_features, n_bond_features


class KEMPNN(nn.Module):
    """KEMPNN with enn, set2set,  knowledge attention
       M=0 -> disable set2set
       T2=0 -> disable knowledge attention (same as MPNN)
       Args:
        T: the number of the iteration of message-passing
        M: the number of the iteration in the set2set
        n_hidden: the number of hidden layers
        n_atom_features: the number of atom features
        n_bond_features: the number of bond features
        T2: the number of the iter. of message-passing in knowledge attention
    """

    def __init__(
        self,
        T=2,
        M=0,
        set2set_layers=2,
        n_hidden=175,
        n_atom_features=n_atom_features,
        n_bond_features=n_bond_features,
        regression=True,
        n_tasks=1,
        T2=1,
    ):
        super().__init__()
        self.n_atom_features = int(n_atom_features)
        self.n_bond_features = int(n_bond_features)
        self.n_hidden = int(n_hidden)
        self.T = int(T)
        self.M = int(M)
        self.set2set_layers = int(set2set_layers)
        self.n_tasks = int(n_tasks)
        self.regression = regression

        self.first = nn.Linear(self.n_atom_features, self.n_hidden)
        self.enn = EdgeNetwork(
            n_features=self.n_bond_features, n_hidden=self.n_hidden
        )
        self.gru = torch.nn.GRUCell(self.n_hidden, self.n_hidden)
        self.atom_embed = nn.Linear(self.n_hidden, self.n_hidden)
        if self.M > 0:
            self.graph_embed = Set2Set(
                M=self.M, n_hidden=self.n_hidden, n_layers=set2set_layers
            )
            self.mlp = nn.Sequential(
                nn.Linear(self.n_hidden * 2, self.n_hidden * 2), nn.ReLU()
            )
            self.out = nn.Linear(self.n_hidden * 2, self.n_tasks)
        else:
            self.graph_embed = AggregateReadout(self.n_hidden)
            self.mlp = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU()
            )
            self.out = nn.Linear(self.n_hidden, self.n_tasks)

        self.T2 = T2
        self.attention_message = Message(self.n_hidden, self.T2)
        self.attention_message_att = Message(
            self.n_hidden, activation="linear"
        )
        self.attention_loss = nn.Sequential(
            nn.Linear(self.n_hidden, 1), nn.Tanh()
        )
        self.attention_readout = AggregateReadout(self.n_hidden)
        self.attention_out = nn.Linear(self.n_hidden, self.n_tasks)

    def forward(
        self,
        node_features,
        edge_features,
        edge_domain,
        edge_range,
        graph_id,
        only_attention=False,
        attention_loss=False,
        for_grad_ram=False,
    ):
        """
            Args:
                node_features: float tensor (n_atoms, n_atom_feature)
                edge_features: float tensor (n_atoms, n_edge_feature)
                edge_domain: int tensor (n_edges, 1)
                edge_range: int tensor (n_edges, 1)
                graph_id: int tensor (n_atoms, 1)
                only_attention: if True returns only knowledge attention value
                attention loss: if True returns knowledge attention
                                value for node regression loss calculation
            Returns:
                out: float tensor (n_molecules, 1)  --  property prediction
                att_out: float tensor (n_molecules, 1)
                        --  property prediction solely by knowledge attention
        """
        h = F.relu(self.first(node_features))

        # message passsing with enn and gru
        for k in range(self.T):
            message = F.relu(
                self.enn(h, edge_features, edge_domain, edge_range)
            )
            h = self.gru(message, h)

        # knowledge attention begins ----
        # message passing in knowledge attention
        att_readout = None
        if self.T2 > 0:
            h_att = h
            for k in range(self.T2):
                message = F.relu(
                    self.enn(h_att, edge_features, edge_domain, edge_range)
                )
                h_att = self.attention_message(message, h_att, k)

            # last layer of attention
            message = F.relu(
                self.enn(h_att, edge_features, edge_domain, edge_range)
            )
            # calculate attention value by linear layer with skip connection
            att = self.attention_message_att(message, h_att)

            if attention_loss:
                # attention value ((-1)-1) for calculating knowledge loss (L_k)
                return self.attention_loss(att)
            if only_attention:
                return att

            # redout for calculating loss (L_kp)
            att_readout = self.attention_readout(message, graph_id)

            h *= att

        # knowledge attention ends ---

        # node embedding
        atom_embedding = h

        # readout
        molecule_embedding = self.graph_embed(atom_embedding, graph_id)

        # fc
        mlp_out = self.mlp(molecule_embedding)

        if for_grad_ram:
            out = self.out(mlp_out)
            return out, atom_embedding

        if self.regression:
            # property prediction
            out = self.out(mlp_out)
            # property prediction by knowledge attention branch (for L_kp)
            if att_readout is not None:
                att_out = self.attention_out(att_readout)
            else:
                att_out = out.new_zeros(out.shape)

            return out.view(-1, self.n_tasks), att_out.view(-1, self.n_tasks)
        else:
            raise NotImplementedError

    def extra_repr(self):
        return (
            "hidden_features={}, T={}, M={}," " set2set_layers={}, T2={}"
        ).format(self.n_hidden, self.T, self.M, self.set2set_layers, self.T2)


class KEMPNNLoss:
    """Loss for KEMPNN
       beta = 0 corresponds to standard MSE Loss for MPNN.

       Args:
           alpha: weight for standard MSE loss (Lp)
           beta: weight for MSE of prediction from knowledge attention (Lkp)

    """

    def __init__(self, alpha=1, beta=0.1):
        self.alpha = alpha
        self.beta = beta
        self.loss = nn.MSELoss()

    def __str__(self):
        return "KEMPNNLoss(alpha={:f}, beta={:f})".format(
            self.alpha, self.beta
        )

    def __call__(self, input, label):
        out, att = input
        return (
            self.loss(out, label) * self.alpha
            + self.loss(att, label) * self.beta
        )


class Message(nn.Module):
    """Message Passing Layers
    """

    def __init__(self, n_hidden, T=1, activation=None):
        super().__init__()
        self.T = T
        self.n_hidden = n_hidden
        self.linear = nn.ModuleList(
            [nn.Linear(self.n_hidden, self.n_hidden) for t in range(self.T)]
        )
        if activation is None:
            self.selu = nn.ReLU()
        elif activation == "sigmoid":
            self.selu = nn.Sigmoid()
        elif activation == "tanh":
            self.selu = nn.Tanh()
        elif activation == "linear":
            self.selu = None
        else:
            raise NotImplementedError

    def forward(self, message, skip, iter=0):
        a = self.linear[iter](message) + skip
        if self.selu is None:
            return a
        h = self.selu(a)
        return h


class EdgeNetwork(nn.Module):
    """Edge network layer
    """

    def __init__(self, n_hidden, n_features):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_features = n_features
        self.E_mlp = nn.Sequential(nn.Linear(n_features, n_hidden), nn.ReLU())
        self.E = nn.Linear(n_hidden * n_hidden, n_hidden, bias=False)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.E.weight)

    def forward(self, node_features, edge_features, edge_domain, edge_range):
        n_nodes = node_features.shape[0]
        mlp_features = self.E_mlp(edge_features)
        A = torch.mm(
            mlp_features, self.E.weight
        )  # n_edge * (n_hidden,n_hidden)
        A = A.reshape((-1, self.n_hidden, self.n_hidden))
        range_node = torch.index_select(
            node_features, 0, edge_range
        ).unsqueeze(2)
        edge_message = torch.bmm(A, range_node).squeeze(2)
        out = edge_message.new_zeros((n_nodes, self.n_hidden))
        out.index_add_(0, edge_domain, edge_message)
        return out


class AggregateReadout(nn.Module):
    """Readout using summation aggregation
    """

    def __init__(self, n_hidden):
        super().__init__()
        self.n_hidden = n_hidden
        self.R = nn.Linear(self.n_hidden, self.n_hidden)

    def forward(self, nodes, graph_id):
        batchsize = torch.max(graph_id) + 1
        atom_embedding = nodes
        # readout
        atom_activation = F.selu_(self.R(atom_embedding))
        graph_sum = atom_activation.new_zeros(
            (batchsize, atom_activation.shape[1])
        )
        graph_sum.index_add_(0, graph_id, atom_activation)
        graph_embedding = torch.tanh_(graph_sum)
        return graph_embedding


class Set2Set(nn.Module):
    """ Readout by set2set layer
    """

    def __init__(self, M, n_hidden, n_layers=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.M = M
        self.LSTM = nn.LSTM(self.n_hidden * 2, self.n_hidden, self.n_layers)
        self.reset_parameters()

    def extra_repr(self):
        return "hidden_features={}, M={} n_layers={}".format(
            self.n_hidden, self.M, self.n_layers
        )

    def forward(self, nodes, graph_id):
        batchsize = int(torch.max(graph_id) + 1)
        c = (
            nodes.new_zeros((self.n_layers, batchsize, self.n_hidden)),
            nodes.new_zeros((self.n_layers, batchsize, self.n_hidden)),
        )
        q = nodes.new_zeros((batchsize, self.n_hidden))
        q_star = None
        for i in range(self.M):
            q_node_length = torch.index_select(q, 0, graph_id)
            e = torch.sum(nodes * q_node_length, 1)
            e_patitioned = dynamic_partition(e, graph_id, batchsize)
            a = torch.cat(
                [F.softmax(es, 0) for es in e_patitioned], 0
            ).unsqueeze(1)
            r = nodes.new_zeros((batchsize, self.n_hidden))
            r = r.index_add_(0, graph_id, a * nodes)
            q_star = torch.cat([q, r], 1)
            if i < self.M - 1:
                q, c = self.LSTM(q_star.unsqueeze(0), c)
                q = q.view(batchsize, self.n_hidden)
        return q_star

    def reset_parameters(self):
        self.LSTM.reset_parameters()


def dynamic_partition(
    data: torch.Tensor, partitions: torch.IntTensor, num_partitions: int
) -> List[torch.Tensor]:
    """Devide the data into partitions using partition indices
    """
    res = []
    for i in range(num_partitions):
        res += [data[(partitions == i).nonzero(as_tuple=False).squeeze(1)]]
    return res


class MPNN(nn.Module):
    """MPNN with ennn, set2set
    """

    def __init__(
        self,
        T=2,
        M=0,
        n_hidden=175,
        n_atom_features=n_atom_features,
        n_bond_features=n_bond_features,
        regression=True,
        n_tasks=1,
    ):
        super().__init__()
        self.n_atom_features = n_atom_features
        self.n_bond_features = n_bond_features
        self.n_hidden = n_hidden
        self.T = T
        self.M = M
        self.n_tasks = n_tasks
        self.regression = regression

        self.message = EdgeNetwork(
            n_features=n_bond_features, n_hidden=n_hidden
        )
        self.update2 = nn.ModuleList(
            [nn.Linear(self.n_hidden, self.n_hidden) for t in range(self.T)]
        )
        self.atom_embed = nn.Linear(self.n_hidden, self.n_hidden)
        if self.M > 0:
            self.graph_embed = Set2Set(
                M=self.M, n_hidden=self.n_hidden, n_layers=M
            )
            self.mlp = nn.Sequential(
                nn.Linear(self.n_hidden * 2, self.n_hidden * 2), nn.ReLU()
            )
            self.out = nn.Linear(self.n_hidden * 2, self.n_tasks)
        else:
            self.graph_embed = AggregateReadout(self.n_hidden)
            self.mlp = nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), nn.ReLU()
            )
            self.out = nn.Linear(self.n_hidden, self.n_tasks)

    def forward(
        self, node_features, edge_features, edge_domain, edge_range, graph_id
    ):
        # zeropad the node feature if node feature is smaller than hidden layer
        if node_features.shape[1] < self.n_hidden:
            padded = node_features.new_zeros(
                (node_features.shape[0], self.n_hidden)
            )
            padded[:, : node_features.shape[1]] = node_features
            node_features = padded

        # message passsing
        h = node_features
        for k in range(self.T):
            message = self.message(h, edge_features, edge_domain, edge_range)
            # h = self.update(message, h)
            h = F.selu_(self.update2[k](message) + h)

        # node embedding
        atom_embedding = self.atom_embed(h)

        # readout
        molecule_embedding = self.graph_embed(atom_embedding, graph_id)

        # fc
        mlp_out = self.mlp(molecule_embedding)

        if self.regression:
            out = self.out(mlp_out)
            return out.view(-1, 1)
        else:
            raise NotImplementedError
