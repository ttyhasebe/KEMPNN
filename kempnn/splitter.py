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
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


def split(x, y, frac_train=0.8, frac_valid=0.1, method="random", seed=None):
    assert method in ["random", "scaffold"]
    if method is None or method == "random":
        return _train_valid_test_split(x, y, frac_train, frac_valid, seed)
    elif method == "scaffold":
        return _scaffold_train_valid_test_split(x, y, frac_train, frac_valid)


def _train_valid_test_split(x, y, frac_train=0.8, frac_valid=0.1, seed=None):
    """ Random Split (mimics DeepChem Implementation)
    """
    if seed is not None:
        np.random.seed(seed)
    n_sample = x.shape[0]
    shuffle = np.random.permutation(range(n_sample))
    n_train = int(n_sample * frac_train)
    n_valid = int(n_sample * (frac_train + frac_valid))

    train = shuffle[:n_train]
    valid = shuffle[n_train:n_valid]
    test = shuffle[n_valid:]
    _all = [train, valid, test]
    ret = [(x[i], y[i]) for i in _all]
    return ret


def _scaffold_train_valid_test_split(x, y, frac_train=0.8, frac_valid=0.1):
    """ Scaffold Split (mimics DeepChem Implementation)
    """
    n_sample = x.shape[0]
    scaffolds = {}
    for i, smiles in enumerate(x):
        scaffold = MurckoScaffoldSmiles(
            mol=Chem.MolFromSmiles(smiles), includeChirality=False
        )
        if scaffold in scaffolds:
            scaffolds[scaffold].append(i)
        else:
            scaffolds[scaffold] = [i]

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set
        for (_, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True
        )
    ]

    train_cutoff = frac_train * n_sample
    valid_cutoff = (frac_train + frac_valid) * n_sample

    train_inds = []
    valid_inds = []
    test_inds = []

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            if (
                len(train_inds) + len(valid_inds) + len(scaffold_set)
                > valid_cutoff
            ):
                test_inds += scaffold_set
            else:
                valid_inds += scaffold_set
        else:
            train_inds += scaffold_set

    _all = [train_inds, valid_inds, test_inds]
    ret = [(x[i], y[i]) for i in _all]
    return ret
