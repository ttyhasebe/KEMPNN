import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from kempnn.loader import MoleculeCollater, MoleculeDataset
from kempnn.models import KEMPNN, AggregateReadout, Set2Set


class TestModel(unittest.TestCase):
    def testKEMPNN(self):
        # toy dataset
        mols = ["COC=O", "OC=O", "CCC=C"]
        a = MoleculeDataset(
            mols, np.array([1, 2, 3]), transform=[StandardScaler()]
        )
        collater = MoleculeCollater(label=True, node_label=False)
        batch = [a[i] for i in range(3)]
        self.data = collater(batch)

        x, y = self.data

        # without set2set
        kempnn = KEMPNN(
            T=2,
            M=0,
            set2set_layers=2,
            n_hidden=175,
            regression=True,
            n_tasks=1,
            T2=1,
        )

        pred_y, pred_y_k = kempnn(*x)

        kempnn(*x, only_attention=True)

        kempnn(*x, attention_loss=True)

    def testKEMPNN_set2set(self):
        # toy dataset
        mols = ["COC=O", "OC=O", "CCC=C"]
        a = MoleculeDataset(
            mols, np.array([1, 2, 3]), transform=[StandardScaler()]
        )
        collater = MoleculeCollater(label=True, node_label=False)
        batch = [a[i] for i in range(3)]
        self.data = collater(batch)

        x, y = self.data
        kempnn = KEMPNN(
            T=2,
            M=2,
            set2set_layers=2,
            n_hidden=175,
            regression=True,
            n_tasks=1,
            T2=1,
        )
        pred_y, pred_y_k = kempnn(*x)

    def testSet2Set(self):
        s2s = Set2Set(2, 100)
        ret = s2s(
            torch.ones((10, 100)).to(torch.float),
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32),
        )
        self.assertEqual(ret.shape[0], 3)

    def testAggregate(self):
        s2s = AggregateReadout(100)
        ret = s2s(
            torch.ones((10, 100)).to(torch.float),
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32),
        )
        self.assertEqual(ret.shape[0], 3)
