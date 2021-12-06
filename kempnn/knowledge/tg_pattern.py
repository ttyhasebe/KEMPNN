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

from kempnn.loader import loadDataset
from kempnn.visualizer import drawSmiles


def make_knowledge_for_tg(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    RotatableBond = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    rotatable = mol.GetSubstructMatches(RotatableBond)
    NonRotatableBond = Chem.MolFromSmarts("[C,c]=,#[C,c]")
    non_rotatable = mol.GetSubstructMatches(NonRotatableBond)

    AromaticBond = Chem.MolFromSmarts("c:c")
    aromaticbond = mol.GetSubstructMatches(AromaticBond)

    isInRing = [
        int(mol.GetAtomWithIdx(i).IsInRing()) for i in range(mol.GetNumAtoms())
    ]

    ret = np.zeros(mol.GetNumAtoms(), dtype=np.float)
    ret += np.array(isInRing) * 1
    for bond in rotatable:
        ret[bond[0]] += -1
        ret[bond[1]] += -1

    for bond in non_rotatable:
        ret[bond[0]] += 0.5
        ret[bond[1]] += 0.5

    for bond in aromaticbond:
        ret[bond[0]] += 1
        ret[bond[1]] += 1

    ret = np.maximum(np.minimum(ret, 1), -1)

    return ret


def load_tg_knowledge():
    knowledge, _, _ = loadDataset(
        dict(
            dataset=dict(
                name="PolymerTg", frac_train=1, frac_test=0, frac_valid=0
            )
        )
    )
    for i in range(len(knowledge)):
        smiles = knowledge[i].smiles
        knowledge.annotate_node(i, make_knowledge_for_tg(smiles))

    return knowledge


if __name__ == "__main__":
    print(make_knowledge_for_tg("[Ce]CC(C)(C(=O)OCCCCCCCC)[Th]"))
    print(make_knowledge_for_tg("[Ce]CC(C1=CC(Cl)=C(C=C1))[Th]"))
    drawSmiles(
        "[Ce]CC(C1=CC(Cl)=C(C=C1))[Th]",
        weights=make_knowledge_for_tg("[Ce]CC(C1=CC(Cl)=C(C=C1))[Th]"),
        filename="test.svg",
    )
