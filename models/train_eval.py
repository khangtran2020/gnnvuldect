import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score


class EarlyStopping:
    def __init__(
        self,
        patience=7,
        mode="max",
        delta=0.001,
        verbose=False,
        run_mode=None,
        skip_ep=100,
    ):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    "EarlyStopping counter: {} out of {}".format(
                        self.counter, self.patience
                    )
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print(
                    "Validation score improved ({} --> {}). Saving model!".format(
                        self.val_score, epoch_score
                    )
                )
            if self.run_mode != "func":
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score


def update_one_batch(model, optimizer, loss_fn, batch, type_of_graph, device) -> float:
    X, Y = batch
    total_loss = 0
    batch_size = len(X)
    for i in range(len(X)):
        data, mask = X[i]
        mask_bin = torch.zeros(data["num_nodes"])
        mask_bin[mask] = 1
        mask_bin = mask_bin.view(-1, 1).to(device)
        for key in type_of_graph:
            if key in data.keys():
                data[key] = data[key].to(device)
        pred = model(data, mask_bin)
        loss = loss_fn(pred, torch.Tensor([Y[i]]).float().to(device))
        total_loss = loss + total_loss
    total_loss = total_loss / batch_size
    total_loss.backward()
    optimizer.step()
    return total_loss.item() * batch_size, batch_size


def update(model, optimizer, loss_fn, loader, type_of_graph, device):
    model.train()
    total_loss = 0
    num_poitns = 0
    print("Training started")
    for i, batch in enumerate(loader):
        optimizer.zero_grad()
        loss, num_pt = update_one_batch(
            model, optimizer, loss_fn, batch, type_of_graph, device
        )
        print(f"Batch {i} Loss: {loss}")
        total_loss = total_loss + loss
        num_poitns = num_poitns + num_pt
    return total_loss / num_poitns


def evaluate_one_batch(model, batch, type_of_graph, device, loss_fn) -> tuple:
    X, Y = batch
    total_loss = 0
    batch_size = len(X)
    preds = []
    for i in range(len(X)):
        data, mask = X[i]
        mask_bin = torch.zeros(data["num_nodes"])
        mask_bin[mask] = 1
        mask_bin = mask_bin.view(-1, 1).to(device)
        for key in type_of_graph:
            if key in data.keys():
                data[key] = data[key].to(device)
        pred = model(data, mask_bin)
        loss = loss_fn(pred, torch.Tensor([Y[i]]).float().to(device))
        preds += pred.to("cpu").detach().numpy().tolist()
        total_loss = loss + total_loss
    total_loss = total_loss / batch_size
    return total_loss.item() * batch_size, batch_size, preds, Y


def evaluate(model, loader, type_of_graph, device, loss_fn):
    model.eval()
    total_loss = 0
    num_points = 0
    all_preds = []
    all_labels = []
    print("Evaluation started")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            loss, num_pt, preds, labels = evaluate_one_batch(
                model, batch, type_of_graph, device, loss_fn
            )
            print(f"Batch {i} Loss: {loss}")
            total_loss = total_loss + loss
            num_points = num_points + num_pt
            all_preds = all_preds + preds
            all_labels = all_labels + labels
    # compute accuracy, auc, f1, precision
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = accuracy_score(all_labels, all_preds.round())
    auc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds.round())
    precision = precision_score(all_labels, all_preds.round())
    return total_loss / num_points, acc, auc, f1, precision
