import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from kempnn.loader import MoleculeCollater, MoleculeDataset, SmilesToGraph


class TestLoader(unittest.TestCase):
    def testSmilesToGraph(self):
        stg = SmilesToGraph()
        instance = stg("COC=O")

        atoms = torch.tensor(
            [
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        )
        edges = torch.tensor(
            [
                [1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
            ]
        )
        edge_range = torch.tensor([1, 2, 3])
        edge_domain = torch.tensor([0, 1, 2])

        self.assertEqual(instance.smiles, "COC=O")
        self.assertEqual(torch.is_tensor(instance.vertices), True)
        self.assertEqual(torch.is_tensor(instance.edges), True)
        self.assertEqual(torch.is_tensor(instance.bond_domain), True)
        self.assertEqual(torch.is_tensor(instance.bond_range), True)
        self.assertEqual(torch.sum((atoms - instance.vertices) ** 2), 0)
        self.assertEqual(torch.sum((edges - instance.edges) ** 2), 0)
        self.assertEqual(torch.sum((edge_range - instance.bond_range) ** 2), 0)
        self.assertEqual(
            torch.sum((edge_domain - instance.bond_domain) ** 2), 0
        )
        self.assertEqual(instance.n_nodes, 4)
        self.assertEqual(instance.n_edges, 3)
        self.assertEqual(instance.label, None)
        self.assertEqual(instance.node_label, None)

    def testEmptyMoleculeDataset(self):
        a = MoleculeDataset([])
        self.assertEqual(a.n_graph, 0)
        self.assertEqual(len(a), 0)

    def testMoleculeDatasetInitialization(self):
        mols = ["COC=O", "OC=O", "CCC=C"]
        a = MoleculeDataset(mols, np.array([1, 2, 3]))
        self.assertEqual(len(a), 3)
        for i in range(3):
            self.assertEqual(a[i].smiles, mols[i])
            self.assertEqual(a[i].label, i + 1)
        a = MoleculeDataset(
            mols,
            None,
            [
                np.array([1, 2, 3, 4]),
                np.array([2, 2, 2]),
                np.array([3, 3, 3, 3]),
            ],
        )
        self.assertEqual(
            (a[0].node_label - torch.tensor([1, 2, 3, 4])).abs().sum(), 0
        )

    def testMoleculeDatasetTransform(self):
        mols = ["COC=O", "OC=O", "CCC=C"]
        a = MoleculeDataset(
            mols, np.array([1, 2, 3]), transform=[StandardScaler()]
        )
        self.assertEqual(a[1].label, 0)
        self.assertEqual(
            (
                (
                    a.inverse_transform(a.y.numpy())
                    - np.array([1, 2, 3]).reshape(-1, 1)
                )
                ** 2
            ).sum(),
            0,
        )

    def testMolecleCollater(self):
        mols = ["COC=O", "OC=O", "CCC=C"]
        a = MoleculeDataset(
            mols, np.array([1, 2, 3]), transform=[StandardScaler()]
        )
        collater = MoleculeCollater(label=True, node_label=False)
        batch = [a[i] for i in range(3)]
        batched = collater(batch)
        (nodes, edges, edge_domain, edge_range, graph_id), ret_y = batched

        self.assertEqual((nodes[0:4] - a[0].vertices).abs().sum(), 0)
        self.assertEqual((nodes[4:7] - a[1].vertices).abs().sum(), 0)
        self.assertEqual((nodes[7:11] - a[2].vertices).abs().sum(), 0)

        self.assertEqual((edges[0:3] - a[0].edges).abs().sum(), 0)
        self.assertEqual((edges[3:5] - a[1].edges).abs().sum(), 0)
        self.assertEqual((edges[5:8] - a[2].edges).abs().sum(), 0)

        self.assertEqual((edges[:8] - edges[8:]).abs().sum(), 0)

        self.assertEqual((edge_domain[0:3] - a[0].bond_domain).abs().sum(), 0)
        self.assertEqual(
            (edge_domain[3:5] - (4 + a[1].bond_domain)).abs().sum(), 0
        )
        self.assertEqual(
            (edge_domain[5:8] - (7 + a[2].bond_domain)).abs().sum(), 0
        )

        self.assertEqual((edge_range[0:3] - a[0].bond_range).abs().sum(), 0)
        self.assertEqual(
            (edge_range[3:5] - (4 + a[1].bond_range)).abs().sum(), 0
        )
        self.assertEqual(
            (edge_range[5:8] - (7 + a[2].bond_range)).abs().sum(), 0
        )

        self.assertEqual((edge_domain[:8] - edge_range[8:]).abs().sum(), 0)
        self.assertEqual((edge_range[:8] - edge_domain[8:]).abs().sum(), 0)

        self.assertEqual((graph_id[:4] - 0).abs().sum(), 0)
        self.assertEqual((graph_id[4:7] - 1).abs().sum(), 0)
        self.assertEqual((graph_id[7:11] - 2).abs().sum(), 0)

        testy = (
            StandardScaler()
            .fit_transform(np.array([1, 2, 3]).reshape((-1, 1)))
            .reshape(-1)
        )
        self.assertLess((ret_y - torch.tensor(testy)).abs().sum(), 1e-7)

    def testMolecleCollaterNodeLabel(self):
        mols = ["COC=O", "OC=O", "CCC=C"]
        node_label = [
            np.array([1, 2, 3, 4]),
            np.array([2, 3, 4]),
            np.array([3, 4, 5, 6]),
        ]

        dataset = MoleculeDataset(mols, None, node_label)
        collater = MoleculeCollater(label=False, node_label=True)
        batch = [dataset[i] for i in range(3)]
        batched = collater(batch)
        (nodes, edges, edge_domain, edge_range, graph_id), node_y = batched

        self.assertEqual(
            (node_y[0:4] - torch.tensor([1, 2, 3, 4])).abs().sum(), 0
        )
        self.assertEqual(
            (node_y[4:7] - torch.tensor([2, 3, 4])).abs().sum(), 0
        )
        self.assertEqual(
            (node_y[7:11] - torch.tensor([3, 4, 5, 6])).abs().sum(), 0
        )
