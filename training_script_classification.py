import logging
import matplotlib.pyplot as plt
import os
import pickle
import seaborn as sns
import torch
import torch.nn.functional as F

from argparse import ArgumentParser, BooleanOptionalAction
from batching_script import customBatching
from data_processing import *
from model_scripts import GAT, SAGE, SAGE_ATTN
from torch_geometric.data import Data
from torchmetrics import AveragePrecision, AUROC, Precision, Recall, F1Score, ConfusionMatrix
from tqdm import tqdm
from typing import Literal, Tuple

##### load data

model_map = {
    "GAT": GAT, 
    "SAGE": SAGE, 
    "SAGE_ATTN": SAGE_ATTN
}

TRAINING_INFO_TEMPLATE = "Epoch: {CURRENT_EPOCH}/{NUM_EPOCHS}; Batch: {CURRENT_BATCH}/{NUM_BATCHES}; Loss: {CURRENT_LOSS:0.4f}"
LOGGING_INFO_TEMPLATE = "Epoch: {CURRENT_EPOCH}/{NUM_EPOCHS}; Loss: {CURRENT_LOSS:0.4f}"
LOGGING_INFO_TEMPLATE_VAL = "Epoch: {CURRENT_EPOCH}/{NUM_EPOCHS}; Loss: {CURRENT_LOSS:0.4f} Validation Loss: {VAL_LOSS:0.4f}"
BATCH_SIZE = 512

def load_data(dataset_name: Literal["elliptic++", "dgraphfin"]="elliptic++", load_test_data=False) -> Tuple[Data, Data, int]:
    match dataset_name:
        case "elliptic++":
            train_data, test_data, OUT_DIM = load_create_ellipticpp(load_test_data=load_test_data, scale_data=False)
        case "dgraphfin":
            # @TODO: UPDATE DGRAPHFIN LOADING
            train_data, test_data, OUT_DIM = load_dgraphfin(load_test_data=load_test_data)

    return train_data, test_data, OUT_DIM

##### train loop

def train_epoch(model: torch.nn.Module, optimizer, train_batches, iterator: tqdm):
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

    torch.cuda.empty_cache()
    
    return running_loss.item()/len(train_batches)

def train_run(model, optimizer, train_batches, valid_batches, do_validation=False):
    loss_all = []
    loss_validation = []

    iterator = tqdm(range(args.epochs))
    early_stopping_handler = earlyStoppingHandler(max_patience=10, threshold=0.05, save_checkpoint=True, path=PATH, max_checkpoints=7, save_interval=10, saving_strategy="periodic")
    for epoch in iterator:
        iterator.set_description(TRAINING_INFO_TEMPLATE.format(
            CURRENT_EPOCH=epoch+1,
            NUM_EPOCHS=args.epochs,
            CURRENT_BATCH=0,
            NUM_BATCHES=len(train_batches),
            CURRENT_LOSS=-1,
            )
        )

        loss = train_epoch(model, optimizer, train_batches, iterator)
        loss_all += [loss]

        if do_validation:
            scores, cm, val_loss = evaluation(valid_batches,model,device="cuda:0", return_loss=True)
            LOGGING_FILE.info(LOGGING_INFO_TEMPLATE_VAL.format(
                CURRENT_EPOCH=epoch+1,
                NUM_EPOCHS=args.epochs,
                CURRENT_LOSS=loss,
                VAL_LOSS=val_loss
                )
            )
            LOGGING_FILE.info(
                f"""
                P={scores[0]:0.4f} | R={scores[1]:0.4f} | F1={scores[2]:0.4f} | AP={scores[3]:0.4f} | AUROC={scores[4]:0.4f}
                """
            )
            loss_validation += [val_loss]

        if early_stopping_handler.check_stopping(loss, epoch, model):
            print("Stopping Early")
            LOGGING_FILE.info("Stopping Early")
            break

    if do_validation:
        return loss_all, loss_validation
    else:
        return loss_all

def evaluation(data_batches, model, device, return_loss=False):
    torch.cuda.empty_cache()
    model.eval()

    cm_torch = ConfusionMatrix(task="binary").to(device)
    p_torch = Precision(task="binary").to(device)
    r_torch = Recall(task="binary").to(device)
    f1_torch = F1Score(task="binary").to(device)
    ap_torch = AveragePrecision(task="binary", thresholds=100).to(device)
    auroc_torch = AUROC(task="binary", thresholds=100).to(device)

    confusion_matrix = torch.zeros((2,2), device=device)
    scores = []
    loss = []
    with torch.no_grad():
        for batch in tqdm(data_batches):
            pred = model(batch.x, batch.edge_index)

            loss += [F.cross_entropy(pred[batch.train_mask], batch.y[batch.train_mask]).item()]

            pred_labels = F.softmax(pred, dim=1)
            orig_labels = batch.y

            confusion_matrix += cm_torch(pred_labels.argmax(dim=1), orig_labels)
            scores += [
                [
                    p_torch(pred_labels.argmax(dim=1), orig_labels),
                    r_torch(pred_labels.argmax(dim=1), orig_labels),
                    f1_torch(pred_labels.argmax(dim=1), orig_labels),
                    ap_torch(pred_labels.max(dim=1)[0], orig_labels),
                    auroc_torch(pred_labels.max(dim=1)[0], orig_labels),
                ]
            ]
    
    scores = torch.as_tensor(scores)

    if not return_loss:
        return scores.mean(dim=0).tolist(), confusion_matrix.tolist()
    else:
        return scores.mean(dim=0).tolist(), confusion_matrix.tolist(), sum(loss)/len(loss)

def display_losses(loss_list, path, display_plot=False):
    sns.set_theme(style="whitegrid")
    labels = ["Training", "Validation"]
    for loss, label in zip(loss_list, labels):
        sns.lineplot(x=range(len(loss)), y=loss, label=label)
    
    plt.legend()
    plt.savefig(os.path.join(path, "training_loss.png"))

    if display_plot:
        plt.show()

class earlyStoppingHandler:
    def __init__(self, max_patience, threshold, save_checkpoint=False, path="", max_checkpoints=7, save_interval=10, saving_strategy: Literal["max", "periodic"]="max"):
        self.max_patience = max_patience
        self.current_patience = 0
        self.threshold_delta = threshold
        self.save_checkpoint = save_checkpoint
        self.max_checkpoints = max_checkpoints
        self.saving_strategy = saving_strategy
        self.save_interval = save_interval

        self.save_path = path

        self.min_loss = float("inf")

    def _save_checkpoint_to_file(self, epoch, model_checkpoint):
        match self.saving_strategy:
            case "max":
                checkpoint_files = [file for file in os.listdir(self.save_path) if file.endswith(".pt")]
                checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.removesuffix(".pt").split("_")[-1]))
                if len(checkpoint_files) > self.max_checkpoints:
                    for model_file in checkpoint_files[:(-self.max_checkpoints+1)]:
                        os.remove(os.path.join(self.save_path, model_file))
                torch.save(model_checkpoint.state_dict(), os.path.join(self.save_path, f"model_checkpoint_{epoch+1}.pt"))
            case "periodic":
                if (epoch+1) % self.save_interval == 0:
                    torch.save(model_checkpoint.state_dict(), os.path.join(self.save_path, f"model_checkpoint_{epoch+1}.pt"))

    def check_stopping(self, current_loss, epoch, model_checkpoint):
        if self.saving_strategy == "periodic":
                self._save_checkpoint_to_file(epoch, model_checkpoint)

        if current_loss < self.min_loss:
            self.min_loss = current_loss
            self.current_patience = 0
            if self.save_checkpoint:
                self._save_checkpoint_to_file(epoch, model_checkpoint)
        elif current_loss - self.threshold_delta > self.min_loss:
            self.current_patience += 1
            if self.current_patience >= self.max_patience:
                return True
        
        return False

if __name__ == "__main__":
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--epochs", help="number of epochs", type=int)
    arg_parser.add_argument("--model", help="model architecture", type=lambda x: x.upper(), choices=["GAT", "GCN", "SAGE", "SAGE_ATTN"])
    arg_parser.add_argument("--dataset", help="dataset to use", type=str, choices=["elliptic++", "dgraphfin"])
    arg_parser.add_argument("--device", help="device to train on", type=str)
    arg_parser.add_argument("--neighbourhood", help="neighbourhood sizes", type=lambda x: [int(num) for num in x[1:-1].split(",")])
    arg_parser.add_argument("--path", help="where to store model training results", type=str)
    arg_parser.add_argument("--lr", help="learning rate in scientific notation", type=float)
    arg_parser.add_argument("--validation", help="do validation testing", type=bool, action=BooleanOptionalAction)

    args = arg_parser.parse_args()
    train_data, test_data, OUT_DIM = load_data(args.dataset, load_test_data=not (args.validation))
    DEVICE = torch.device(args.device)

    PATH = args.path

    ### @TODO: CONSIDER FULL NEIGHBOURHOOD FOR GAT?? (BCS SAMPLING IS MAINLY DONE BY SAGE)
    train_batches = customBatching(train_data, positive_label=1, negative_label=0, neighbourhood_sizes=args.neighbourhood, batch_size=BATCH_SIZE, device=DEVICE)
    test_batches = customBatching(test_data, positive_label=1, negative_label=0, neighbourhood_sizes=[-1] * len(args.neighbourhood), batch_size=BATCH_SIZE, device=DEVICE)
    
    model_parameters = {
        "conv_size": 256,
        "mlp_size": 512
    }

    model_config = model_map[args.model]
    model = model_config(
        in_dim=next(iter(train_batches)).x.shape[-1],
        out_dim=OUT_DIM,
        **model_parameters
    ).to(DEVICE) # instantiate model with hyperparameters
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    
    LOGGING_FILE = logging.getLogger(args.model)
    logging.basicConfig(filename=os.path.join(PATH, "training_log.log"), level=logging.INFO, force=True)
    
    LOGGING_FILE.info(f"Doing validation: {args.validation}" + "\n------------------------\n")
    LOGGING_FILE.info(str(model) + "\n------------------------\n")
    LOGGING_FILE.info(str(optimizer) + "\n------------------------\n")
    LOGGING_FILE.info(f"Number of parameters: {str(model.get_num_parameters())}" + "\n------------------------\n")
    LOGGING_FILE.info(f"Neighbourhood sizes: {str(args.neighbourhood)}" + "\n------------------------\n")

    if args.validation:
        loss_all, loss_validation = train_run(model, optimizer, train_batches, test_batches, do_validation=args.validation)
        loss_display = [loss_all, loss_validation]
    else:
        loss_all = train_run(model, optimizer, train_batches, test_batches, do_validation=args.validation)
        loss_display = [loss_all]

    display_losses(loss_list=loss_display, path=PATH, display_plot=False)

    model_checkpoint_files = sorted([file for file in os.listdir(PATH) if file.endswith(".pt")], key=lambda x:int(x.removesuffix(".pt").split("_")[-1]))
    for model_checkpoint_file in model_checkpoint_files:
        torch.cuda.empty_cache()
        model = model_config(
            in_dim=next(iter(train_batches)).x.shape[-1],
            out_dim=OUT_DIM,
            **model_parameters
        ).to(DEVICE) # instantiate model with hyperparameters
        model.load_state_dict(torch.load(os.path.join(PATH, model_checkpoint_file), weights_only=True))
        model.eval()

        eval_scores, confusion_matrix = evaluation(test_batches, model, device=DEVICE)
        LOGGING_FILE.info(f"----------{model_checkpoint_file}---------")
        LOGGING_FILE.info(
            f"""
            TN={confusion_matrix[0][0]}\t|FP=\t{confusion_matrix[0][1]}
            --------------------
            FN={confusion_matrix[1][0]}\t|TP=\t{confusion_matrix[1][1]}
            """
        )
        LOGGING_FILE.info(
            f"""
            Precision={eval_scores[0]}\n
            Recall={eval_scores[1]}\n
            F1={eval_scores[2]}\n
            AP/AUPRC={eval_scores[3]}\n
            AUROC={eval_scores[4]}\n
            """
        )

        pickle.dump(eval_scores, open(os.path.join(PATH, f"eval_scores_{model_checkpoint_file}.pickle"), "wb"))
        pickle.dump(confusion_matrix, open(os.path.join(PATH, f"confusion_matrix_{model_checkpoint_file}.pickle"), "wb"))

    torch.save(model.state_dict(), os.path.join(PATH, f"model_checkpoint_FINAL.pt"))
    pickle.dump(loss_all, open(os.path.join(PATH, "training_loss.pickle"), "wb"))
