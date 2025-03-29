from typing import Literal, Tuple
from torch_geometric.data import Data
from data_processing import *
from tqdm import tqdm
from argparse import ArgumentParser
from batching_script import customBatching
import torch
import torch.nn.functional as F
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from model_scripts import GAT, GCN, SAGE, SAGE_ATTN
##### load data

model_map = {
    "GAT": GAT, 
    "GCN": GCN, 
    "SAGE": SAGE, 
    "SAGE_ATTN": SAGE_ATTN
}

TRAINING_INFO_TEMPLATE = "Epoch: {CURRENT_EPOCH}/{NUM_EPOCHS}; Batch: {CURRENT_BATCH}/{NUM_BATCHES}; Loss: {CURRENT_LOSS:0.4f}"

def load_data(dataset_name: Literal["elliptic++", "dgraphfin"]="elliptic++") -> Tuple[Data, Data, int]:
    match dataset_name:
        case "elliptic++":
            train_data, valid_data, OUT_DIM = load_create_ellipticpp(load_test_data=False)
        case "dgraphfin":
            # @TODO: UPDATE DGRAPHFIN LOADING
            train_data, valid_data, OUT_DIM = load_dgraphfin(load_test_data=False)

    return train_data, valid_data, OUT_DIM

##### train loop

def train_epoch(model: torch.nn.Module, optimizer, train_batches, valid_batches, iterator: tqdm):
    model.train()
    running_loss = 0
    epoch_info, _ = iterator.desc.split("; ")[:-1]
    epoch_info = epoch_info.split(": ")[-1].split("/")
    for idx, batch in enumerate(train_batches):
        out = model(batch.x, batch.edge_index)

        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss

        iterator.set_description(TRAINING_INFO_TEMPLATE.format(
            CURRENT_EPOCH=epoch_info[0],
            NUM_EPOCHS=epoch_info[1],
            CURRENT_BATCH=idx+1,
            NUM_BATCHES=len(train_batches),
            CURRENT_LOSS=running_loss.item()/(idx+1)
            )
        )

    # @TODO: do validation
    # @TODO: add logging
    model.eval()
    runing_valid_loss = 0
    for idx, batch in enumerate(valid_batches):
        out = model(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        runing_valid_loss += loss

    print(f"Validation loss: {runing_valid_loss.item()/len(valid_batches):0.4f}")

    return running_loss.item()/len(train_batches), runing_valid_loss.item()/len(valid_batches)

def display_losses(train_loss, valid_loss, path):
    sns.set_theme(style="whitegrid")

    sns.lineplot(x=range(len(train_loss)), y=train_loss, label="Training")
    sns.lineplot(x=range(len(train_loss)), y=valid_loss, label="Validation")
    
    plt.legend()
    plt.savefig(os.path.join(path, "training_validation_loss.png"))
    plt.show()

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--epochs", help="number of epochs", type=int)
    arg_parser.add_argument("--model", help="model architecture", type=lambda x: x.upper(), choices=["GAT", "GCN", "SAGE", "SAGE_ATTN"])
    arg_parser.add_argument("--dataset", help="dataset to use", type=str, choices=["elliptic++", "dgraphfin"])
    arg_parser.add_argument("--device", help="device to train on", type=str)
    arg_parser.add_argument("--neighbourhood", help="neighbourhood sizes", type=lambda x: [int(num) for num in x.split(",")])

    args = arg_parser.parse_args()
    train_data, valid_data, OUT_DIM = load_data(args.dataset)
    DEVICE = torch.device(args.device)
    model_checkpoint_path = f"model_checkpoints/{args.model}"
    os.makedirs(model_checkpoint_path, exist_ok=True)

    train_batches = customBatching(train_data, positive_label=1, negative_label=0, neighbourhood_sizes=args.neighbourhood, batch_size=64, device=DEVICE)
    valid_batches = customBatching(valid_data, positive_label=1, negative_label=0, neighbourhood_sizes=[-1] * len(args.neighbourhood), batch_size=64, device=DEVICE)
    
    model_parameters = {
        "hidden_size": 64
    }

    model = model_map[args.model]
    model = model(
        in_dim=next(iter(train_batches)).x.shape[-1],
        out_dim=OUT_DIM,
        **model_parameters
    ).to(DEVICE) # instantiate model with hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-3)

    loss_all = []
    valid_loss_all = []

    max_patience = 5
    current_patience = 0
    min_loss = -1
    iterator = tqdm(range(args.epochs))
    for epoch in iterator:
        iterator.set_description(TRAINING_INFO_TEMPLATE.format(
            CURRENT_EPOCH=epoch+1,
            NUM_EPOCHS=args.epochs,
            CURRENT_BATCH=0,
            NUM_BATCHES=len(train_batches),
            CURRENT_LOSS=-1,
            )
        )

        loss, valid_loss = train_epoch(model, optimizer, train_batches, valid_batches, iterator)
        loss_all += [loss]
        valid_loss_all += [valid_loss]

        if valid_loss < min_loss or epoch < 1:
            min_loss = valid_loss
            current_patience = 0
        elif current_patience < max_patience:
            current_patience += 1
        if current_patience >= max_patience:
            print("Stopping early")
            # @TODO: save model checkpoint
            break

    display_losses(train_loss=loss_all, valid_loss=valid_loss_all, path=model_checkpoint_path)

    torch.save(model.state_dict(), os.path.join(model_checkpoint_path, f"{args.model}_checkpoint.pt"))
    pickle.dump(loss_all, open(os.path.join(model_checkpoint_path, "training_loss.pickle"), "wb"))
    pickle.dump(valid_loss_all, open(os.path.join(model_checkpoint_path, "validation_loss.pickle"), "wb"))
