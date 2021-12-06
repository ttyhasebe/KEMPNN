# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import time

import numpy as np
import torch
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from tqdm import tqdm

from .loader import loadDataset
from .trainer import ConfigEncoder, MoleculeTrainer


def mean_std(arr):
    return np.mean(arr), np.std(arr)


def runSingle(cfg):
    """Evaluate the model accuracy with a single run.
    """
    return runEvaluation(cfg, 1, 1)


def runEvaluation(cfg, split_runs=3, model_runs=3, filename="result"):
    """Evaluate the model accuracy with multiple runs.
        Args:
            cfg(dict): configuration data
            split_runs(int): the number of runs for different
                             dataset-split random seeds.
            model_runs(int): the number of runs for different
                             model-initialization random seeds.
            filename(string): a file-name for logging
    """
    seeds = [i + 1 for i in range(split_runs)]
    model_seeds = [i + 10 for i in range(model_runs)]
    datasets = []
    for s in seeds:
        if split_runs != 1:
            cfg["dataset"]["seed"] = s
        d = loadDataset(cfg)
        datasets.append(d)

    trainer = MoleculeTrainer()

    results = []

    if "knowledge" in cfg and cfg["knowledge"] is not None:
        knowledge = cfg["knowledge"]["dataset"]["data"]
        trainer.setKnowledgeDataset(knowledge)

    for split_seed, dataset in tqdm(zip(seeds, datasets)):
        # set random seed for model
        cfg["dataset"]["seed"] = split_seed
        trainer.setDataset(*dataset)
        for seed in model_seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = cfg["model"](**cfg["model_arg"])

            result = trainer.fit(model, cfg)
            del model
            result["split_seed"] = split_seed
            result["model_seed"] = seed
            result["cfg"] = cfg
            results.append(result)

    test_rmse = mean_std([r["test_rmse"] for r in results])
    val_rmse = mean_std([r["val_rmse"] for r in results])
    best_test_rmse = mean_std([r["best_test_rmse"] for r in results])
    best_val_rmse = mean_std([r["best_val_rmse"] for r in results])
    summary = {
        "test_rmse": test_rmse,
        "val_rmse": val_rmse,
        "best_test_rmse": best_test_rmse,
        "best_val_rmse": best_val_rmse,
        "results": results,
    }

    summary_json = ConfigEncoder.dumps(summary)
    with open(f"{filename}.json", "w") as fp:
        fp.write(summary_json)
    print(summary)
    return summary


def runOptimization(
    cfg,
    optimize_cfg,
    n_iter=20,
    split_runs=1,
    model_runs=1,
    filename="optimize_result",
):
    """Optimize the model parameter using hyperopt.
    The model parameters are optimized using
    the evaluations on validation dataset.
        Args:
            cfg(dict): configuration data
            optimize_cfg(dict): configuration for optimization
            n_iter(int): the number of iterations for sequential optimization
            split_runs(int): the number of runs
                            for different dataset-split random seeds.
            model_runs(int): the number of runs
                            for different model-initialization random seeds.
            filename(string): a file-name for logging
    """

    def objective(space):
        print(space)
        newcfg = {**cfg}
        for k in space.keys():
            if k in newcfg and type(newcfg[k]) == dict:
                newcfg[k] = {**space[k]}
            else:
                newcfg[k] = space[k]

        print(newcfg, cfg)
        result = runEvaluation(
            newcfg, split_runs=split_runs, model_runs=model_runs
        )

        opt_result = {
            "loss": result["val_rmse"][0],
            "loss_variance": result["val_rmse"][1] ** 2,
            "true_loss": result["test_rmse"][0],
            "true_loss_variance": result["test_rmse"][1] ** 2,
            "status": STATUS_OK,
            "eval_time": time.time(),
            "data": result,
            "space": space,
        }
        return opt_result

    trials = Trials()

    best = fmin(
        objective,
        optimize_cfg,
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=trials,
    )

    valid_trial = [t for t in trials if t["result"]["status"] == STATUS_OK]
    losses_argmin = np.argmin(
        [float(trial["result"]["loss"]) for trial in valid_trial]
    )
    print([float(trial["result"]["loss"]) for trial in valid_trial])
    best_trial = valid_trial[losses_argmin]
    best_result = best_trial["result"]["data"]
    print(best, best_trial["result"]["space"], space_eval(optimize_cfg, best))

    ret = {
        "best": best,
        "n_iter": n_iter,
        "split_runs": split_runs,
        "model_runs": model_runs,
        "result": best_result,
        "optimize_confg": optimize_cfg,
        "config": cfg,
    }
    ret_str = ConfigEncoder.dumps(ret)
    with open(f"{filename}.json", "w") as fp:
        fp.write(ret_str)

    print(ret)

    return ret
