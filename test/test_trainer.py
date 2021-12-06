import re
import unittest
from contextlib import redirect_stdout
from io import StringIO

import torch

from kempnn.knowledge import load_esol_crippen_knowledge
from kempnn.loader import loadESOL
from kempnn.models import KEMPNN, KEMPNNLoss
from kempnn.trainer import MoleculeTrainer, defaultMoleculeTrainConfig


class TestTrainer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = loadESOL()
        self.knowledge = load_esol_crippen_knowledge()

    def testTrainer(self):
        trainer = MoleculeTrainer()
        trainer.setDataset(*self.dataset)
        trainer.setKnowledgeDataset(self.knowledge)

        model = KEMPNN()

        # The following config variable species hyperparameters for training.
        training_config = dict(**defaultMoleculeTrainConfig)
        training_config.update(
            name="test",
            loss=KEMPNNLoss(1, 0.1),  # loss weight for L_p, L_kp
            optimizer=torch.optim.Adam,
            optimizer_args={"lr": 0.00065},
            optimize_schedule=torch.optim.lr_scheduler.StepLR,
            optimize_schedule_args={"step_size": 75, "gamma": 0.95},
            epochs=2,
            batch_size=16,
            save=False,
            save_path="weights",
            knowledge=dict(
                pretrain_epoch=2,
                train_factor=0.2,  # loss weight for L_k
                loss=torch.nn.MSELoss(),
                optimizer=torch.optim.SGD,
                optimizer_args_pretrain={"momentum": 0.9, "lr": 0.01},
                batch_size=32,
            ),
        )
        with StringIO() as buf, redirect_stdout(buf):
            trainer.fit(model, training_config)
            output = buf.getvalue()
            self.assertRegex(
                output, r"knowledge_pretrain.*epoch:2.*rmse:[0-9\.]+"
            )
            self.assertRegex(output, r"Training result:.*test_rmse:[0-9\.]+")

    def testTrainerWoKnowledge(self):
        trainer = MoleculeTrainer()
        trainer.setDataset(*self.dataset)
        trainer.setKnowledgeDataset(self.knowledge)

        model = KEMPNN()

        # The following config variable species hyperparameters for training.
        training_config = dict(**defaultMoleculeTrainConfig)
        training_config.update(
            name="test",
            loss=KEMPNNLoss(1, 0.123),  # loss weight for L_p, L_kp
            optimizer=torch.optim.SGD,
            optimizer_args={"lr": 0.000123},
            optimize_schedule=torch.optim.lr_scheduler.StepLR,
            optimize_schedule_args={"step_size": 75, "gamma": 0.95},
            epochs=2,
            batch_size=10,
            save=False,
            save_path="weights",
            knowledge=None,
        )
        with StringIO() as buf, redirect_stdout(buf):
            ret, dbg = trainer.fit(model, training_config, debug=True)
            output = buf.getvalue()
            self.assertEqual(
                re.match(
                    r"knowledge_pretrain.*epoch:2.*rmse:[0-9\.]+", output
                ),
                None,
            )
            self.assertRegex(output, r"Training result:.*test_rmse:[0-9\.]+")
            self.assertEqual(type(dbg["optimizer"]).__name__, "SGD")
            self.assertEqual(dbg["loss_func"].beta, 0.123)
            self.assertEqual(dbg["optimizer"].param_groups[0]["lr"], 0.000123)
            self.assertEqual(type(dbg["scheduler"]).__name__, "StepLR")
            self.assertEqual(dbg["epochs"], 2)
            self.assertEqual(dbg["batch_size"], 10)
