#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import numpy as np
from rdkit import Chem


def oneHot(x, allowable_set):
    if x not in allowable_set:
        return list(map(lambda _: 0, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def oneHotUnique(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


n_atom_features = 37
n_bond_features = 10


def atom_features(atom, explicitH=False):
    results = (
        oneHotUnique(
            atom.GetSymbol(),
            [
                "C",
                "N",
                "O",
                "S",
                "F",
                "Si",
                "P",
                "Cl",
                "Br",
                "As",
                "I",
                "B",
                "Se",
                "Te",
                "At",
            ],
        )
        + oneHot(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + oneHot(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        )
        + [atom.GetIsAromatic()]
    )

    if not explicitH:
        results = results + oneHot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    try:
        results = (
            results
            + oneHot(atom.GetProp("_CIPCode"), ["R", "S"])
            + [atom.HasProp("_ChiralityPossible")]
        )
    except Exception:
        results = (
            results + [False, False] + [atom.HasProp("_ChiralityPossible")]
        )

    return np.array(results)


def bond_features(bond):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    bond_feats = bond_feats + oneHot(
        str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
    )

    return np.array(bond_feats, dtype=np.int16)
