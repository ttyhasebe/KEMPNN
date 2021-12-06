#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import torch

from .loader import SmilesToGraph
from .utils import drawSmiles


def MPNNGradRAM(mpnn, graph):
    mpnn.zero_grad()

    output = mpnn(*graph)
    gradient = torch.cat(torch.autograd.grad(output, mpnn.graph_feature))

    with torch.no_grad():
        alpha = torch.mean(gradient, dim=0)
        grad_cam = torch.sum(mpnn.graph_feature * alpha, dim=1)
    return grad_cam, output


def MPNNGradRAMFromSmiles(mpnn, smiles, show=False, filename=None):
    x = SmilesToGraph()(smiles)
    grad_cam, output = MPNNGradRAM(mpnn, x)
    if show:
        drawSmiles(smiles, weights=grad_cam.cpu().numpy(), filename=filename)
    return grad_cam, output
