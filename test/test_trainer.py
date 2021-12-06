import unittest

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from kempnn.loader import MoleculeCollater, MoleculeDataset, loadESOL
from kempnn.models import KEMPNN, KEMPNNLoss
from kempnn.trainer import MoleculeTrainer, defaultMoleculeTrainConfig
from kempnn.knowledge import knowledgeDatasetFromFunc, load_esol_crippen_knowledge
from kempnn.knowledge.crippen_pattern import make_crippen_knowledge_attention
from io import StringIO
import re
from contextlib import redirect_stdout
import sys

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
            optimize_schedule=torch.optim.lr_scheduler.MultiStepLR,
            optimize_schedule_args={"milestones": [75, 100, 125], "gamma": 0.95},
            epochs=2,
            batch_size=16,
            save=False,
            save_path="weights",
            knowledge=dict(
                pretrain_epoch=2,  # epochs of knowledge pretain,0 -> disable pretrain
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
            self.assertRegex(output, r'knowledge_pretrain.*epoch:2.*rmse:[0-9\.]+')
            self.assertRegex(output, r'Training result:.*test_rmse:[0-9\.]+')

    def testTrainerWoKnowledge(self):
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
            optimize_schedule=torch.optim.lr_scheduler.MultiStepLR,
            optimize_schedule_args={"milestones": [75, 100, 125], "gamma": 0.95},
            epochs=2,
            batch_size=16,
            save=False,
            save_path="weights",
            knowledge=None
        )
        with StringIO() as buf, redirect_stdout(buf):
            trainer.fit(model, training_config)
            output = buf.getvalue()
            self.assertEqual(re.match(r'knowledge_pretrain.*epoch:2.*rmse:[0-9\.]+', output), None)
            self.assertRegex(output, r'Training result:.*test_rmse:[0-9\.]+')

