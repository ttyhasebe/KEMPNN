import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from kempnn import (
    KEMPNN,
    KEMPNNLoss,
    MoleculeDataset,
    MoleculeTrainer,
    defaultMoleculeTrainConfig,
)
from kempnn.knowledge.crippen_pattern import make_crippen_knowledge_attention
from kempnn.loader import loadMoleculeDataset

# This file is example of the KEMPNN program for custom datasets and trainer

# Define your own dataset path
csv_dataset_path = (
    "../dataset/MoleculeNet/ESOL.csv"
)  # path to your csv dataset
task_name = (
    "measured log solubility in mols per litre"
)  # the name of target property column.
smiles_name = "smiles"  # the name of smiles column

# Define your own knowledge for knowledge learning
# smiles_for_knowledge_annotation = ["COCOC", "C=CC", ... ] # array of smiles
# knowledge_annotation_labels =
#                    [np.array([1,0,1,0,1]), ...] # array of ndarray(n_atoms)
data = pd.read_csv("../dataset/MoleculeNet/ESOL.csv")
smiles_for_knowledge_annotation = data["smiles"]  # array of smiles
knowledge_annotation_labels = [
    make_crippen_knowledge_attention(s) for s in data["smiles"]
]  # array of ndarray(n_atoms): per node annotation of konwledge

knowledge = MoleculeDataset(
    smiles_for_knowledge_annotation, node_label=knowledge_annotation_labels
)


dataset = loadMoleculeDataset(
    data["smiles"],
    data["measured log solubility in mols per litre"],
    frac_train=0.8,
    frac_test=0.1,
    frac_valid=0.1,
    seed=1,
    method="random",
    transform=[StandardScaler()],
)

trainer = MoleculeTrainer()
trainer.setDataset(*dataset)
trainer.setKnowledgeDataset(knowledge)

model = KEMPNN()

# The following config variable species hyperparameters for training.
training_config = dict(**defaultMoleculeTrainConfig)
training_config.update(
    name="example",
    loss=KEMPNNLoss(1, 0.1),  # loss weight for L_p, L_kp
    optimizer=torch.optim.Adam,
    optimizer_args={"lr": 0.00065},
    optimize_schedule=torch.optim.lr_scheduler.MultiStepLR,
    optimize_schedule_args={"milestones": [75, 100, 125], "gamma": 0.95},
    epochs=150,
    batch_size=16,
    save=True,  # Save model weights and logs
    save_path="weights",
    knowledge=dict(
        pretrain_epoch=30,  # epochs of knowledge pretain,0 -> disable pretrain
        train_factor=0.2,  # loss weight for L_k
        loss=torch.nn.MSELoss(),
        optimizer=torch.optim.SGD,
        optimizer_args_pretrain={"momentum": 0.9, "lr": 0.01},
        batch_size=32,
    ),
)

trainer.fit(model, training_config)
