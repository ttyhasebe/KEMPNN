#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .loader import MoleculeCollater, MoleculeDataset
from .models import KEMPNN


class Visualizer:
    def __init__(self, log_path, model_args={}, dataset=None):
        self.log_path = log_path
        json_path = os.path.join(log_path, "config.json")
        with open(json_path) as fp:
            self.cfg = json.load(fp)

        if dataset is not None:
            assert dataset.cfg["seed"] == self.cfg["train_dataset"]["seed"]
            assert dataset.cfg["method"] == self.cfg["train_dataset"]["method"]
            assert dataset.cfg["name"] == self.cfg["train_dataset"]["name"]
            self.dataset = dataset
        else:
            self.dataset = None

        model_path = os.path.join(log_path, "best_model.pth")

        with open(os.path.join(self.log_path, "transform.pkl"), "rb") as fp:
            self.transform = pickle.load(fp)
        print(model_path)
        if os.path.exists(model_path):
            self.model = KEMPNN(**model_args)
            self.model.load_state_dict(torch.load(model_path))

    def train_curve(self, plot_test=True):
        losses = (
            torch.load(os.path.join(self.log_path, "losses.pth")).cpu().numpy()
        )
        x = np.arange(losses.shape[1])
        plt.plot(x, losses[0, :], label="train")
        if plot_test:
            plt.plot(x, losses[1, :], label="test")
        plt.plot(x, losses[2, :], label="valid")
        plt.legend()
        plt.xlabel("# of epochs")
        plt.ylabel("MSE loss")
        plt.tight_layout()
        plt.show()

    def smiles_examples(self, smiles, labels, filename):
        labels = np.array(labels)
        data = MoleculeDataset(
            smiles, labels, transform=self.transform, train=False
        )
        print(self.transform)
        self.model.eval()
        dataloader = torch.utils.data.DataLoader(
            data,
            batch_size=1,
            shuffle=False,
            collate_fn=MoleculeCollater(label=True),
            pin_memory=True,
            drop_last=False,
        )

        attentions = []
        y_pred_value_inv = []
        y_inv = []
        for batch in dataloader:
            self.model.zero_grad()
            x, y = batch
            y_pred_value, atom_embedding = self.model(*x, for_grad_ram=True)

            gradient = torch.cat(
                torch.autograd.grad(y_pred_value, atom_embedding)
            )
            with torch.no_grad():
                alpha = torch.mean(gradient, dim=0)
                grad_cam = torch.sum(atom_embedding * alpha, dim=1)

            y_pred_value_inv.append(
                data.inverse_transform(y_pred_value.detach())
            )
            y_inv.append(data.inverse_transform(labels))
            attentions.append(grad_cam)
        print(attentions)
        print(y_pred_value_inv)
        print(y_inv)

        import svgutils.transform as sg

        # create new SVG figure
        n_col = 5
        n_row = (len(data) - 1) // 5 + 1
        unit = 300
        unit_y = 200
        fig = sg.SVGFigure(unit * n_col, n_row * unit_y)
        attentions_scale = torch.max(
            torch.tensor([torch.max(torch.abs(a)) for a in attentions])
        ).numpy()
        for i in range(len(data)):
            _x = (i % 5) * 300
            _y = (i // 5) * 200
            sm = data[i].smiles
            sm = sm.replace("[Ce]", "[*]").replace("[Th]", "[*]")
            chem_svg = drawSmiles(
                sm,
                weights=attentions[i].numpy() / attentions_scale,
                size=(unit, unit_y),
            )
            fig1 = sg.fromstring(chem_svg)
            plot1 = fig1.getroot()
            plot1.moveto(_x, _y)
            fig.append([plot1])
            txt1 = sg.TextElement(
                25 + _x,
                20 + _y,
                "pred={:.3f}, true={:.3f}".format(
                    y_pred_value_inv[i][0, 0], y_inv[0][i, 0]
                ),
                size=14,
                weight="bold",
            )
            fig.append(txt1)

        fig.save(filename)


def drawSmiles(smiles, size=(300, 300), weights=None, filename=None):
    m = Chem.MolFromSmiles(smiles)
    view = rdMolDraw2D.MolDraw2DSVG(*size)
    tm = rdMolDraw2D.PrepareMolForDrawing(m)
    if weights is not None:
        atoms = [a.GetIdx() for a in m.GetAtoms()]
        colors = {}
        radius = {}
        for i, a in enumerate(atoms):
            color = plt.cm.bwr_r(np.clip(weights[i] * 0.3 + 0.5, 0.2, 0.8))
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
    return svg
