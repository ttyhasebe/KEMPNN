import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from kempnn import (
    KEMPNN,
    KEMPNNLoss,
    defaultMoleculeTrainConfig,
    runEvaluation,
)
from kempnn.knowledge.crippen_pattern import make_crippen_knowledge_attention
from kempnn.loader import MoleculeDataset

# This file is example of the KEMPNN program for custom datasets

# Define your own dataset path
csv_dataset_path = "dataset/MoleculeNet/ESOL.csv"  # path to your csv dataset
task_name = (
    "measured log solubility in mols per litre"
)  # the name of target property column.
smiles_name = "smiles"  # the name of smiles column

# Define your own knowledge for knowledge learning
# smiles_for_knowledge_annotation = ["COCOC", "C=CC", ... ] # array of smiles
# knowledge_annotation_labels = [np.array([1,0,1,0,1]),
#                         np.array([1,0,0]), ...] # array of ndarray(n_atoms)
data = pd.read_csv("../dataset/MoleculeNet/ESOL.csv")
smiles_for_knowledge_annotation = data["smiles"]  # array of smiles
knowledge_annotation_labels = [
    make_crippen_knowledge_attention(s) for s in data["smiles"]
]  # array of ndarray(n_atoms): per node annotation of konwledge

knowledge = MoleculeDataset(
    smiles_for_knowledge_annotation, node_label=knowledge_annotation_labels
)


# The following config variable species ML models,
# hyperparameters and other settings.

config = dict(**defaultMoleculeTrainConfig)
config.update(
    name="example",
    model=KEMPNN,  # ML model
    model_arg=dict(),  # arguments for ml model (dictionary)
    dataset=dict(
        name="custom_dataset",
        seed=1,
        frac_train=0.8,
        frac_test=0.1,
        frac_valid=0.1,
        method="random",
        transform=[StandardScaler()],  # list of scalers
        data_path=csv_dataset_path,  # path to csv dataset
        csv_dataset=True,
        task_name=task_name,  # header name of the target property
        smiles_name=smiles_name,  # header name of smiles in the csv file
    ),
    loss=KEMPNNLoss(1, 0.1),  # loss weight for L_p, L_kp
    optimizer=torch.optim.Adam,
    optimizer_args={"lr": 0.00065},
    optimize_schedule=torch.optim.lr_scheduler.MultiStepLR,
    optimize_schedule_args={"milestones": [75, 100, 125], "gamma": 0.95},
    epochs=2,
    batch_size=16,
    save=True,  # Save model weights and logs
    save_path="weights",
    knowledge=dict(
        dataset=dict(name="custom_knowledge", data=knowledge),
        pretrain_epoch=2,  # epochs for knoelwdge pretain, 0=disable pretrain
        train_factor=0.2,  # loss weight for L_k
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.SGD,
        optimizer_args_pretrain={"momentum": 0.9, "lr": 0.01},
        batch_size=32,
    ),
)

# Execute traning and model evaluation using the configulation.
runEvaluation(config, split_runs=1, model_runs=1, filename="result")
