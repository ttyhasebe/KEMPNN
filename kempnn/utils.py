#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import sys
import time
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from scipy.stats import pearsonr


def peason_r2_score(y: np.ndarray, y_pred: np.ndarray) -> float:
    if (y - y_pred).var() < 1e-10:
        return np.nan
    return pearsonr(y, y_pred)[0] ** 2


def rmse_score(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2, axis=0))


def mse_score(y, y_pred):
    return np.sqrt(np.mean((y - y_pred) ** 2, axis=0))


start_time = 0


def download(url, filename):
    def reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s"
            % (percent, progress_size / (1024 * 1024), speed)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, filename, reporthook)


def drawSmiles(smiles, size=(300, 300), weights=None, filename=None):
    m = Chem.MolFromSmiles(smiles)
    view = rdMolDraw2D.MolDraw2DSVG(*size)
    tm = rdMolDraw2D.PrepareMolForDrawing(m)

    if weights is not None:
        atoms = [a.GetIdx() for a in m.GetAtoms()]
        colors = {}
        radius = {}
        for i, a in enumerate(atoms):
            color = plt.cm.bwr_r(np.clip(weights[i] * 0.2 + 0.5, 0.3, 0.7))
            colors[a] = color
            radius[a] = 0.5
        view.DrawMolecule(
            tm,
            highlightAtoms=atoms,
            highlightBonds={},
            highlightAtomColors=colors,
            highlightAtomRadii=radius,
        )
    else:
        view.DrawMolecule(tm)

    view.FinishDrawing()
    svg = view.GetDrawingText()
    if filename:
        with open(filename, "w") as f:
            f.write(svg)
