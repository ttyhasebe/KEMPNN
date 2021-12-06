#
# Copyright 2021 by Tatsuya Hasebe, Hitachi, Ltd.
# All rights reserved.
#
# This file is part of the KEMPNN package,
# and is released under the "BSD 3-Clause License". Please see the LICENSE
# file that should have been included as part of this package.
#

import datetime
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.utils.data

from .loader import MoleculeCollater, loadDataset
from .utils import peason_r2_score, rmse_score

defaultMoleculeTrainConfig = {
    "name": "",
    "device": "cuda",
    "optimizer": torch.optim.Adam,
    "optimizer_args": {"lr": 0.001},
    "optimize_schedule": None,
    "optimize_schedule_args": {},
    "loss": torch.nn.MSELoss(),
    "save": True,
    "save_path": "weights",
    "batch_size": 16,
    "epochs": 50,
    "drop_last": True,
}


class ConfigEncoder(json.JSONEncoder):
    # overload method default
    def default(self, obj):

        # Match all the types you want to handle in your converter
        if isinstance(obj, (float, int, str, dict, list, tuple)):
            return json.JSONEncoder.default(self, obj)

        if hasattr(obj, "__class__"):
            if obj.__class__.__name__ == "type":
                return obj.__name__  # Call the default method for other type
            return str(obj)  # Call the default method for other type

        return json.JSONEncoder.default(self, obj)

    @classmethod
    def dumps(cls, obj):
        return json.dumps(obj, indent=4, cls=cls)


class MoleculeTrainer:
    """ Train molecule dataset
    """

    def __init__(self):
        self.default_cfg = defaultMoleculeTrainConfig
        self.trained = False
        self.dataset = None
        self.att_dataset = None
        pass

    def setDataset(self, train_dataset, test_dataset, valid_dataset):
        """Set training, test, validation dataset
        """
        self.dataset = (train_dataset, test_dataset, valid_dataset)

    def setKnowledgeDataset(self, data):
        """Set dataset for knolwedge learning (molecule dataset with node_label)
        """
        self.att_dataset = data

    def prepareData(self, cfg):
        self.dataset = loadDataset(cfg)

    def fit(self, model, cfg=None, verbose=True):
        """Execute model traning.
        """
        if cfg is None:
            cfg = self.default_cfg

        assert self.dataset is not None

        # dataset
        train_dataset, test_dataset, valid_dataset = self.dataset

        # send model to device
        device = cfg["device"]
        model.to(device)

        # configure save path and save the configurations
        model_dir = ""
        if "dataset" in cfg and "name" in cfg["dataset"]:
            name = cfg["name"] + "_" + cfg["dataset"]["name"]
        else:
            name = cfg["name"]

        root_save_path = cfg["save_path"]
        save = cfg["save"]
        print_text = None
        if save:
            model_dir = os.path.join(
                root_save_path,
                name
                + "_"
                + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
            )
            os.makedirs(model_dir, exist_ok=True)

            cfg["model_path"] = model_dir
            cfg["model_str"] = str(model)

            with open(model_dir + "/config.json", "w") as fp:
                fp.write(ConfigEncoder.dumps(cfg))

            with open(model_dir + "/transform.pkl", "wb") as fp:
                pickle.dump(train_dataset.transform, fp)

            print_text = open(model_dir + "/output.log", "w")

        # define SGD optimizer and its schedule
        optimizer = cfg["optimizer"](
            model.parameters(), **cfg["optimizer_args"]
        )

        if cfg["optimize_schedule"] is not None:
            scheduler = cfg["optimize_schedule"](
                optimizer, **cfg["optimize_schedule_args"]
            )
        else:
            scheduler = None

        # number of epoches
        n_epoch = cfg["epochs"]

        # define dataloader using batch_size
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            collate_fn=MoleculeCollater(label=True),
            pin_memory=True,
            drop_last=cfg["drop_last"],
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=MoleculeCollater(label=True),
            pin_memory=True,
            drop_last=False,
        )

        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            collate_fn=MoleculeCollater(label=True),
            pin_memory=True,
            drop_last=False,
        )

        # define dataloader for knowledge data
        use_knowledge = False
        if "knowledge" in cfg and cfg["knowledge"] is not None:
            assert self.att_dataset is not None
            use_knowledge = True
            att_dataloader = torch.utils.data.DataLoader(
                self.att_dataset,
                batch_size=cfg["knowledge"]["batch_size"],
                shuffle=True,
                collate_fn=MoleculeCollater(label=False, node_label=True),
                pin_memory=True,
                drop_last=True,
            )
        else:
            att_dataloader = None

        # define loss
        loss_func = cfg["loss"]

        # define variables used in traning
        train_loss_log = torch.zeros(n_epoch).to(device)
        test_loss_log = torch.zeros(n_epoch).to(device)
        val_loss_log = torch.zeros(n_epoch).to(device)

        n_train = len(train_dataset)
        n_test = len(test_dataset)
        n_val = len(valid_dataset)
        n_batch = n_train // cfg["batch_size"]
        n_batch_test = n_test // cfg["batch_size"]
        n_batch_val = n_val // cfg["batch_size"]

        best_valid_rmse = 1e20
        best_test_rmse = 1e20
        best_epoch = -1
        test_rmse = None
        val_rmse = None

        # define optimizer and loss for knowledge training
        if use_knowledge:
            k_cfg = cfg["knowledge"]
            k_optimizer = k_cfg["optimizer"](
                model.parameters(), **k_cfg["optimizer_args_pretrain"]
            )
            k_loss_func = k_cfg["loss"]
            if "optimize_schedule" in k_cfg:
                k_scheduler = k_cfg["optimize_schedule"](
                    k_optimizer, **k_cfg["optimize_schedule_args"]
                )
            else:
                k_scheduler = None
        else:
            k_cfg = None
            k_optimizer = None
            k_loss_func = None
            k_scheduler = None

        # execute knowledge pre-training if configured.
        if use_knowledge and cfg["knowledge"]["pretrain_epoch"] > 0:
            assert self.att_dataset is not None
            k_pre_loss_log = torch.zeros(
                cfg["knowledge"]["pretrain_epoch"]
            ).to(device)
            k_n_batch = len(self.att_dataset) // cfg["knowledge"]["batch_size"]
            k_n_epoch = cfg["knowledge"]["pretrain_epoch"]
            for epoch in range(k_n_epoch):
                start_time = time.time()
                model.train()

                # batch learning
                for batch in att_dataloader:
                    x, y = batch
                    # send to gpu
                    x = [_x.to(device) for _x in x]
                    y = y.to(device)

                    k_optimizer.zero_grad()
                    y_pred = model(*x, attention_loss=True)
                    loss = k_loss_func(y_pred.view(-1, 1), y.view(-1, 1))
                    loss.backward()
                    with torch.no_grad():
                        k_pre_loss_log[epoch] += loss / k_n_batch
                    k_optimizer.step()

                if k_scheduler:
                    k_scheduler.step()
                # batch evaluation
                print(
                    f"knowledge_pretrain epoch:{epoch + 1}/{k_n_epoch}"
                    f" rmse:{torch.sqrt(k_pre_loss_log[epoch]):.4f}"
                )

        use_knowledge_train = (
            use_knowledge and cfg["knowledge"]["train_factor"] > 0
        )

        # batch learning for traning dataset
        for epoch in range(n_epoch):
            start_time = time.time()
            model.train()

            if device == "cuda":
                torch.cuda.empty_cache()

            # iterate batch
            for batch in train_dataloader:
                optimizer.zero_grad()

                # calculate knowledge loss (\gamma L_k)
                knowledge_loss = 0
                if use_knowledge_train:
                    k_batch = next(iter(att_dataloader))
                    k_x, k_y = k_batch
                    # send to gpu
                    k_x = [_x.to(device) for _x in k_x]
                    k_y = k_y.to(device)

                    k_y_pred = model(*k_x, attention_loss=True)
                    knowledge_loss = (
                        k_loss_func(k_y_pred.view(-1, 1), k_y.view(-1, 1))
                        * cfg["knowledge"]["train_factor"]
                    )

                # calculate loss (L_p + \gamma_kp L_kp)
                x, y = batch
                # send to gpu
                x = [_x.to(device) for _x in x]
                y = y.to(device)
                y_pred = model(*x)
                loss = loss_func(y_pred, y.view(-1, 1))

                # add knowledge loss
                if use_knowledge_train:
                    loss += knowledge_loss
                with torch.no_grad():
                    train_loss_log[epoch] += loss / n_batch

                loss.backward()
                optimizer.step()

            if scheduler:
                scheduler.step()

            # batch evaluation
            model.eval()
            y_test_all = []
            y_pred_test_all = []
            y_val_all = []
            y_pred_val_all = []

            # evaluate on test set
            with torch.no_grad():
                for batch in test_dataloader:
                    x, y_val = batch
                    # send to gpu
                    x = [_x.to(device) for _x in x]
                    y_val = y_val.to(device)

                    y_pred_val = model(*x)
                    test_loss_log[epoch] += (
                        loss_func(y_pred_val, y_val.view(-1, 1)) / n_batch_test
                    )

                    # record label for r2 calculation
                    y_test_all.append(y_val.cpu().numpy())
                    if type(y_pred_val) == tuple:
                        y_pred_test_all.append(
                            y_pred_val[0][:, 0].cpu().numpy()
                        )
                    else:
                        y_pred_test_all.append(y_pred_val[:, 0].cpu().numpy())

            # evaluate on validation set
            with torch.no_grad():
                for batch in valid_dataloader:
                    x, y_val = batch
                    # send to gpu
                    x = [_x.to(device) for _x in x]
                    y_val = y_val.to(device)

                    y_pred_val = model(*x)
                    val_loss_log[epoch] += (
                        loss_func(y_pred_val, y_val.view(-1, 1)) / n_batch_val
                    )

                    # record label for r2 calculation
                    y_val_all.append(y_val.cpu().numpy())
                    if type(y_pred_val) == tuple:
                        y_pred_val_all.append(
                            y_pred_val[0][:, 0].cpu().numpy()
                        )
                    else:
                        y_pred_val_all.append(y_pred_val[:, 0].cpu().numpy())

            # calulate metrics
            # inverse-transform the properties to
            # evaluate metrics in the original scale.
            y_test_all_inv = test_dataset.inverse_transform(
                np.concatenate(y_test_all)
            )[:, 0]
            y_pred_test_all_inv = test_dataset.inverse_transform(
                np.concatenate(y_pred_test_all)
            )[:, 0]
            y_val_all_inv = valid_dataset.inverse_transform(
                np.concatenate(y_val_all)
            )[:, 0]
            y_pred_val_all_inv = valid_dataset.inverse_transform(
                np.concatenate(y_pred_val_all)
            )[:, 0]

            test_rmse = rmse_score(y_test_all_inv, y_pred_test_all_inv)
            val_rmse = rmse_score(y_val_all_inv, y_pred_val_all_inv)
            try:
                test_r2 = peason_r2_score(y_test_all_inv, y_pred_test_all_inv)
            except ValueError:
                test_r2 = np.nan
            try:
                val_r2 = peason_r2_score(y_val_all_inv, y_pred_val_all_inv)
            except ValueError:
                val_r2 = np.nan

            train_loss_ = train_loss_log.cpu().numpy()[epoch]
            test_loss_ = test_loss_log.cpu().numpy()[epoch]
            val_loss_ = val_loss_log.cpu().numpy()[epoch]

            # save and print result
            if best_valid_rmse > val_rmse:
                if save:
                    torch.save(
                        model.state_dict(), model_dir + "/best_model.pth"
                    )
                best_valid_rmse = val_rmse
                best_test_rmse = test_rmse
                best_epoch = epoch + 1

            text = (
                f"epoch {epoch+1:d}/{n_epoch:d} "
                f"train_loss: {train_loss_:.4f} test_loss: {test_loss_:.4f} "
                f"test_r2: {test_r2:.4f} test_rmse: {test_rmse:.4f} "
                f"val_loss: {val_loss_:.4f} val_r2: {val_r2:.4f} "
                f"val_rmse: {val_rmse:.4f} "
                f"time: {time.time() - start_time:.2f}sec"
            )

            if verbose:
                print(text)
            if save:
                print_text.write(text + "\n")

        if save:
            torch.save(
                torch.stack((train_loss_log, test_loss_log, val_loss_log)),
                model_dir + "/losses.pth",
            )
            torch.save(model, model_dir + "/last_model.pth")
            print_text.close()

        self.trained = True
        ret = {
            "test_rmse": test_rmse,
            "val_rmse": val_rmse,
            "best_test_rmse": best_test_rmse,
            "best_val_rmse": best_valid_rmse,
            "epoch": best_epoch,
            "model_dir": model_dir,
        }

        print(
            f"Training result: "
            f"test_rmse:{test_rmse:.5f} val_rmse:{val_rmse:.5f}\n"
            f"best_epoch:{best_epoch} best_test_rmse:{best_test_rmse:.5f} "
            f"best_val_rmse:{best_valid_rmse:.5f}"
        )
        return ret
