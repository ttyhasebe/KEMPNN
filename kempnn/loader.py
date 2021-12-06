#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import os
from typing import Any, Collection, Dict, List, NamedTuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import StandardScaler

from .features import atom_features, bond_features
from .splitter import split
from .utils import download

# default path of dataset file
data_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "dataset")
)


class GraphInstance:
    """ Structure of graph of single molecule.

    Members:
        smiles(str): smiles string.
        vertices(tensor(n_atoms, n_atom_features)): node feature vectors.
        edges(tensor(n_edges, n_edges_features)): edge feature vectors.
        bond_domain(tensor(n_edges, 1)): atom indices of domain of edge
        bond_range(tensor(n_edges, 1)): atom indices of range of edge
        n_nodes(int): number of nodes
        n_edges(int): number of edges
        label(float, Tensor): target value
        node_label(tensor(n_nodes, :)): per node annotation values
    """

    def __init__(
        self,
        smiles: str,
        vertices: np.ndarray,
        edges: np.ndarray,
        bond_domain: np.ndarray,
        bond_range: np.ndarray,
        label: Union[None, float] = None,
    ):
        self.smiles = smiles
        self.vertices = torch.as_tensor(vertices).to(torch.float)
        self.edges = torch.as_tensor(edges).to(torch.float)
        self.bond_domain = torch.as_tensor(bond_domain).to(torch.long)
        self.bond_range = torch.as_tensor(bond_range).to(torch.long)
        self.n_nodes = self.vertices.shape[0]
        self.n_edges = self.edges.shape[0]
        self.label: Union[torch.Tensor, float, None] = label
        self.node_label: Union[torch.Tensor, None] = None

    def __str__(self):
        return "GraphInstanece(" + self.smiles + ")"


class SmilesToGraph:
    """ Transform smiles string into graph structure.
    """

    def __init__(self, atom_func=atom_features, bond_func=bond_features):
        self.atom_func = atom_func
        self.bond_func = bond_func

    def __call__(self, smiles: str) -> GraphInstance:
        """ Returns GraphInstance from smiles string
        """
        mol = Chem.MolFromSmiles(smiles)  # : Rdkit
        atoms = list(mol.GetAtoms())
        atoms_idx = [a.GetIdx() for a in atoms]

        bonds = list(mol.GetBonds())
        idx = {idx: i for i, idx in enumerate(atoms_idx)}
        bond_domain = np.array(
            [idx[b.GetBeginAtomIdx()] for b in bonds], dtype=np.int32
        )
        bond_range = np.array(
            [idx[b.GetEndAtomIdx()] for b in bonds], dtype=np.int32
        )

        vertices = np.array([self.atom_func(a) for a in atoms])
        edges = np.array([self.bond_func(a) for a in bonds])

        return GraphInstance(smiles, vertices, edges, bond_domain, bond_range)


def atleast2d(array):
    if len(array.shape) < 2:
        return array.reshape((-1, 1))
    return array


class MoleculeDataset(torch.utils.data.Dataset):
    """Define dataset of molecules,
    their target property values and node annotation values(knowledge).
    """

    def __init__(
        self,
        smiles_array: Collection[str],
        y: Union[np.ndarray, None] = None,
        node_label: Union[List[np.ndarray], None] = None,
        transform: List[Any] = [],
        edge_feature: bool = True,
        train: bool = True,
        cfg: Union[Dict, None] = None,
    ):
        """Initialize dataset
            Args:
                smiles_array(string[n_graph]): smiles strings
                y(ndarray(n_graph, :)): target property values
                node_label(list(ndarray(n_nodes, :))): node annotation values
                transform: list of target value transformers
                                        with scikit learn interface.
                train: If ture, train the transform
                                        based on the "y" argument value.
                cfg: configulation data (just for caching).
        """
        if cfg:
            self.cfg = cfg
        else:
            self.cfg = {}
        if len(smiles_array) == 0:
            self.n_graph = 0
            self.graphs = []
            self.y = []
            return

        smiles_transform = SmilesToGraph()

        graphs = [smiles_transform(s) for s in smiles_array]
        self.graphs = graphs
        self.n_graph = len(graphs)

        assert self.n_graph > 0

        self.n_vertex_feat = graphs[0].vertices.shape[1]
        for g in graphs:
            if g.n_edges > 0:
                self.n_edge_feat = g.edges.shape[1]
                break
            self.n_edge_feat = 0

        assert self.n_edge_feat > 0

        if node_label is not None:
            for g, nl in zip(self.graphs, node_label):
                g.node_label = torch.as_tensor(nl).to(torch.float)

        if y is not None:
            self.transform = transform
            if transform is None or len(transform) == 0:
                y = atleast2d(y)
            else:
                if train:
                    for t in self.transform:
                        if y is not None:
                            y = t.fit_transform(atleast2d(y))
                else:
                    for t in self.transform:
                        if y is not None:
                            y = t.transform(atleast2d(y))

            self.y = torch.as_tensor(y).to(torch.float)
        else:
            self.transform = []
            self.y = None

    def __len__(self):
        return self.n_graph

    def __getitem__(self, idx: Any):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        g = self.graphs[idx]
        if self.y is not None:
            g.label = self.y[idx, 0]

        return g

    def annotate_node(self, idx, value):
        self.graphs[idx].node_label = torch.as_tensor(value).to(torch.float)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        for t in self.transform[::-1]:
            y = t.inverse_transform(atleast2d(y))
        return y


class MoleculeGraphBatch(NamedTuple):
    nodes: torch.Tensor
    edges: torch.Tensor
    edge_domain: torch.Tensor
    edge_range: torch.Tensor
    graph_id: torch.Tensor


class MoleculeCollater:
    """Collate the multiple molecule graphs into single array for batch learing.
    """

    def __init__(self, label=True, node_label=False):
        """Initialize molecule collater
        Args:
            label: if true, batch the property data
            node_label: if true, batch the node annotation data (knowledge)
        """
        self.label = label
        self.node_label = node_label

    def __call__(self, batch):
        """Collate the batch of graph instances.
        Args:
        batch(List(GraphInstance)): batched molecule graph instance

        Returns:
        (nodes, edges, edge_domain, edge_range, graph_id), ret_y
            where,
            * nodes is concatenated node feature array.
            * edges is concatenated edge feature array.
            * edge_domain is concatenated edges' domin indices array.
            * edge_range is concatenated edges' range indices array.
            * graph_id(tensor(n_nodes)) is array of integer
                       to identify a molecule that a node belongs to.
            * ret_y (tensor(n_graphs, n_tasks)) is array of molecular property.

        """
        nodes = torch.cat([b.vertices for b in batch], 0)
        edges = torch.cat([b.edges for b in batch], 0)
        bond_domain = torch.cat([b.bond_domain for b in batch], 0)
        bond_range = torch.cat([b.bond_range for b in batch], 0)

        n_nodes = [b.n_nodes for b in batch]
        n_edges = [b.n_edges for b in batch]

        # graph_id
        graph_id = torch.zeros(nodes.shape[0], dtype=torch.long)

        node_count = 0
        edge_count = 0
        for i in range(len(batch)):
            graph_id[node_count : node_count + n_nodes[i]] = i
            bond_domain[edge_count : edge_count + n_edges[i]] += node_count
            bond_range[edge_count : edge_count + n_edges[i]] += node_count
            node_count += n_nodes[i]
            edge_count += n_edges[i]

        # make graph symmetric
        edge_domain = torch.cat([bond_domain, bond_range], 0)
        edge_range = torch.cat([bond_range, bond_domain], 0)
        edges = torch.cat([edges, edges], 0)

        ret_y = None
        if self.label:
            ret_y = torch.stack([b.label for b in batch])

        if self.node_label:
            ret_node = torch.cat([b.node_label for b in batch], 0)
            return (
                MoleculeGraphBatch(
                    nodes, edges, edge_domain, edge_range, graph_id
                ),
                ret_node,
            )

        return (
            MoleculeGraphBatch(
                nodes, edges, edge_domain, edge_range, graph_id
            ),
            ret_y,
        )


# configulation for loading dataset
defaultDatasetConfig = {
    "name": "",  # name of dataset
    "seed": 123,  # random seed for splitting dataset
    "frac_train": 0.8,  # frac. train for splitting
    "frac_valid": 0.1,  # frac. valid for splitting
    "frac_test": 0.1,  # frac. test for splitting
    "method": "random",  # method for splitting
    "transform": None,  # transform algorithm for property data
    "data_path": None,  # path for csv dataset
    "csv_dataset": True,
    "task_name": None,  # header name of the target property in the csv file
    "smiles_name": None,  # header name of smiles in the csv file
}

# configulation template for datasets that extends default config
datasetTemplate = {
    "ESOL": {
        "data_path": os.path.join(data_dir, "MoleculeNet/ESOL.csv"),
        "csv_dataset": True,
        "task_name": "measured log solubility in mols per litre",
        "smiles_name": "smiles",
        "transform": [StandardScaler()],
        "method": "random",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",  # noqa: E501
        "url_post_process": None,
    },
    "FreeSolv": {
        "data_path": os.path.join(data_dir, "MoleculeNet/FreeSolv.csv"),
        "csv_dataset": True,
        "task_name": "expt",
        "smiles_name": "smiles",
        "transform": [StandardScaler()],
        "method": "random",
        "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",  # noqa: E501
        "url_post_process": None,
    },
    "Lipop": {
        "data_path": os.path.join(data_dir, "MoleculeNet/Lipophilicity.csv"),
        "csv_dataset": True,
        "task_name": "exp",
        "smiles_name": "smiles",
        "transform": [StandardScaler()],
        "method": "random",
        "url": "http://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv",  # noqa: E501
        "url_post_process": None,
    },
    "PolymerTg": {
        "data_path": os.path.join(data_dir, "Polymer/Tg.csv"),
        "csv_dataset": True,
        "task_name": "Experiment Tg (K)",
        "smiles_name": "smiles",
        "transform": [StandardScaler()],
        "method": "random",
        "dropna": True,
        "url": os.path.join(data_dir, "ap0c00524_si_001.xlsx"),
        "url_post_process": "tg",
    },
}


def loadDataset(cfg: Dict = {}):
    """Create MoleculeDataset from configulation data
    """
    assert "dataset" in cfg
    assert (
        cfg["dataset"]["name"] != "" or cfg["dataset"]["data_path"] is not None
    )

    newcfg = {}
    newcfg.update(**defaultDatasetConfig)
    if cfg["dataset"]["name"] in datasetTemplate:
        template = datasetTemplate[cfg["dataset"]["name"]]
        newcfg.update(**template)

    newcfg.update(**cfg["dataset"])
    cfg["dataset"] = newcfg
    if newcfg["csv_dataset"]:
        data_path = newcfg["data_path"]
        if not os.path.exists(data_path):
            downloadDataset(
                data_path, newcfg["url"], newcfg["url_post_process"]
            )

        df = pd.read_csv(newcfg["data_path"])
        smiles = df[newcfg["smiles_name"]].values
        EXPR = df[newcfg["task_name"]].values
        nonnan = ~np.isnan(EXPR)
        smiles = smiles[nonnan]
        EXPR = EXPR[nonnan]
        return loadMoleculeDataset(
            smiles,
            EXPR,
            newcfg["frac_train"],
            newcfg["frac_valid"],
            seed=newcfg["seed"],
            method=newcfg["method"],
            transform=newcfg["transform"],
        )


def downloadDataset(data_path, url, postprocess=None):
    if url is None:
        return
    if not os.path.exists(data_path):
        base = os.path.dirname(data_path)
        os.makedirs(base, exist_ok=True)
        if postprocess is None:
            print(f"downloading from {url}")
            download(url, data_path)
        else:
            if postprocess == "tg":
                if not os.path.exists(url):
                    print(
                        "Please download dataset file (XLSX) manually"
                        "from the supporting"
                        "information of the referenced paper[Afzal+ 2021]"
                        "https://pubs.acs.org/doi/10.1021/acsapm.0c00524"
                    )
                    print(
                        "and place the downloaded xlsx file in "
                        '"dataset/ap0c00524_si_001.xlsx".'
                    )
                    import sys

                    sys.exit(0)
                postprocessTg(url, data_path)
            else:
                raise NotImplementedError


def loadMoleculeDataset(
    smiles,
    label,
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=123,
    method="random",
    transform=[],
):
    """ Create MoleculeDataset for train, test,
    validation dataset by the split information
    """
    smiles = np.array(smiles)
    label = np.array(label)
    splitted_data = split(
        smiles, label, frac_train, frac_valid, seed=seed, method=method
    )
    train_dataset = MoleculeDataset(
        splitted_data[0][0],
        splitted_data[0][1],
        transform=transform,
        train=True,
    )
    valid_dataset = MoleculeDataset(
        splitted_data[1][0],
        splitted_data[1][1],
        transform=train_dataset.transform,
        train=False,
    )
    test_dataset = MoleculeDataset(
        splitted_data[2][0],
        splitted_data[2][1],
        transform=train_dataset.transform,
        train=False,
    )
    return [train_dataset, test_dataset, valid_dataset]


def loadESOL(
    frac_train=0.8,
    frac_valid=0.1,
    frac_test=0.1,
    seed=123,
    method="random",
    transform=[StandardScaler()],
):
    dpath = os.path.join(data_dir, "MoleculeNet/ESOL.csv")
    downloadDataset(dpath, datasetTemplate["ESOL"]["url"])
    df = pd.read_csv(dpath)
    smiles = df["smiles"].values
    EXPR = df["measured log solubility in mols per litre"].values
    nonnan = ~np.isnan(EXPR)
    smiles = smiles[nonnan]
    EXPR = EXPR[nonnan]
    return loadMoleculeDataset(
        smiles,
        EXPR,
        frac_train,
        frac_valid,
        seed=seed,
        method=method,
        transform=transform,
    )


def loadFreeSolv(
    frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=123, method="random"
):
    dpath = os.path.join(data_dir, "MoleculeNet/FreeSolv.csv")
    downloadDataset(dpath, datasetTemplate["FreeSolv"]["url"])
    df = pd.read_csv(dpath)
    smiles = df["smiles"].values
    EXPR = df["expt"].values
    nonnan = ~np.isnan(EXPR)
    smiles = smiles[nonnan]
    EXPR = EXPR[nonnan]
    return loadMoleculeDataset(
        smiles, EXPR, frac_train, frac_valid, seed=seed, method=method
    )


def postprocessTg(tmp_filename, out_filename):
    data = pd.read_excel(tmp_filename)
    smiles = data[
        "SMILES (Atoms Ce and Th are placeholders"
        "for head and tail information, respectively)"
    ]
    smiles = [
        s.replace("[[Th]]", "[Th]").replace("[[Ce]]", "[Ce]") for s in smiles
    ]
    data["smiles"] = smiles
    data.to_csv(out_filename)
