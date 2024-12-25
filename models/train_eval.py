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


def update_clean(model, optimizer, objective, batch):
    optimizer.zero_grad()
    input_nodes, output_nodes, mfgs = batch
    inputs = mfgs[0].srcdata["feat"]
    labels = mfgs[-1].dstdata["label"]
    predictions = model(mfgs, inputs)
    # print(f"Size of predictions: {predictions.size()}, Size of labels: {labels.size()}")
    loss = objective(predictions, labels)
    loss.backward()
    optimizer.step()
    return labels, predictions, loss
