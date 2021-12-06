#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import argparse

import numpy as np
import torch
from hyperopt import hp

from kempnn import (
    KEMPNN,
    KEMPNNLoss,
    defaultMoleculeTrainConfig,
    runEvaluation,
    runOptimization,
)
from kempnn.knowledge import load_esol_crippen_knowledge, load_tg_knowledge

parser = argparse.ArgumentParser(
    description="Train/Eval KEMPNN(Knowledge-Embedded MPNN)"
)

parser.add_argument(
    "dataset",
    help="Dataset name",
    type=str,
    choices=["ESOL", "FreeSolv", "Lipop", "PolymerTg"],
)

parser.add_argument(
    "--frac_train", help="Frac. of training split", type=float, default=0.8
)

parser.add_argument(
    "--no_knowledge",
    action="store_true",
    help="Use standard MPNN, not to use KEPMNN with knowledge training",
)
parser.add_argument(
    "--attention",
    action="store_true",
    help="Use KEMPNN without knowledge training"
    " even if the knowledge training is disabled.",
)
parser.add_argument(
    "--no_set2set",
    action="store_true",
    help="Set2set aggregation will not be used.",
)

parser.add_argument(
    "--single",
    action="store_true",
    help="execute only single run without parameter optimization",
)

parser.add_argument(
    "--n_optim",
    type=int,
    default=30,
    help="the number of iterations of hyperparameter optimization",
)
parser.add_argument(
    "--split_runs",
    type=int,
    default=5,
    help="the number of different splits used for evaluation.",
)
parser.add_argument(
    "--model_runs",
    type=int,
    default=3,
    help="the number of evaluation runs "
    "(with different weight initialization) for each split.",
)

parser.add_argument(
    "--save", action="store_true", help="save the model weights"
)
parser.add_argument(
    "--save_path",
    type=str,
    default="weights",
    help="path to save trained neural network weights",
)

parser.add_argument(
    "--postfix",
    type=str,
    default="",
    help="a postfix of the json file name reported.",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.no_knowledge:
        if args.dataset == "PolymerTg":
            knowledge = load_tg_knowledge()
            knowledge_name = "tg_knowledge"
        else:
            knowledge = load_esol_crippen_knowledge()
            knowledge_name = "esol_crippen_knowledge"
    else:
        knowledge = None
        knowledge_name = ""

    save_path = args.save_path

    def make_config(
        dataset, frac_train=0.2, use_knowledge=True, use_attention=False
    ):
        frac_valid = 0.1
        frac_test = 1 - frac_train - frac_valid
        config = dict(**defaultMoleculeTrainConfig)
        config.update(
            name="test",
            model=KEMPNN,
            model_arg=dict(),
            dataset=dict(
                name=dataset,
                frac_train=frac_train,
                frac_test=frac_test,
                frac_valid=frac_valid,
            ),
            loss=KEMPNNLoss(1, 0.1),  # loss weight for L_p, L_kp
            optimizer=torch.optim.Adam,
            optimizer_args={"lr": 0.00065},
            optimize_schedule=torch.optim.lr_scheduler.MultiStepLR,
            optimize_schedule_args={
                "milestones": [75, 100, 125],
                "gamma": 0.95,
            },
            epochs=150,
            batch_size=16,
            save=args.save,
            save_path=save_path,
            knowledge=dict(
                dataset=dict(name=knowledge_name, data=knowledge),
                pretrain_epoch=30,
                train_factor=0.2,  # loss weight for L_k
                loss=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD,
                optimizer_args_pretrain={"momentum": 0.9, "lr": 0.01},
                batch_size=32,
            ),
        )

        if not use_knowledge:
            config["model_arg"]["T2"] = 0
            config["knowledge"] = None
            config["loss"] = KEMPNNLoss(1, 0)

            if use_attention:
                config["loss"] = KEMPNNLoss(1, 0.1)
                config["model_arg"]["T2"] = 1

        return config

    def optimize_eval(
        dataset, frac_train, use_knowledge, optimize_cfg, n_iter, use_attention
    ):
        aa = "kmpnn"
        ua = ""
        if not use_knowledge:
            aa = "mpnn"
            if use_attention:
                ua = "_attention"

        if n_iter > 0:
            optimization_result = runOptimization(
                make_config(dataset, frac_train, use_knowledge, use_attention),
                optimize_cfg,
                n_iter=n_iter,
                filename="expt_{}_{}_{}{}{}_opt".format(
                    dataset, frac_train, aa, ua, args.postfix
                ),
            )
            optimized_cfg = optimization_result["result"]["results"][0]["cfg"]
        else:
            optimized_cfg = make_config(
                dataset, frac_train, use_knowledge, use_attention
            )
        runEvaluation(
            optimized_cfg,
            split_runs=args.split_runs,
            model_runs=args.model_runs,
            filename="expt_{}_{}_{}{}{}_eval".format(
                dataset, frac_train, aa, ua, args.postfix
            ),
        )

    optimize_cfg_no_set2set = dict(
        model_arg={
            "M": 0,
            "T": hp.quniform("T", 2, 6, 1),
            "n_hidden": hp.quniform("n_hidden", 50, 300, 1),
        },
        optimizer_args={
            "lr": hp.loguniform("lr", np.log(0.00001), np.log(0.01))
        },
        optimize_schedule_args={
            "milestones": [75, 100, 125],
            "gamma": hp.uniform("lr_gamma", 0.4, 1),
        },
    )

    optimize_cfg_set2set = dict(
        model_arg=hp.choice(
            "aggregate",
            [
                {
                    "M": 0,
                    "T": hp.quniform("T", 2, 6, 1),
                    "n_hidden": hp.quniform("n_hidden", 50, 300, 1),
                },
                {
                    "M": hp.quniform("M", 1, 6, 1),
                    "T": hp.quniform("T_", 2, 6, 1),
                    "n_hidden": hp.quniform("n_hidden_", 50, 300, 1),
                },
            ],
        ),
        optimizer_args={
            "lr": hp.loguniform("lr", np.log(0.00001), np.log(0.01))
        },
        optimize_schedule_args={
            "milestones": [75, 100, 125],
            "gamma": hp.uniform("lr_gamma", 0.4, 1),
        },
    )

    optimize_cfg = optimize_cfg_set2set
    if args.no_set2set:
        optimize_cfg = optimize_cfg_no_set2set

    if args.single:
        args.n_optim = 0
        args.split_runs = 1
        args.model_runs = 1

    optimize_eval(
        args.dataset,
        args.frac_train,
        not args.no_knowledge,
        optimize_cfg,
        args.n_optim,
        args.attention,
    )
