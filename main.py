import os
import wandb
import torch
import datetime
import numpy as np
from config import parse_args
from data.utils import get_data, custom_collate
from utils.console import console
from utils.utils import seed_everything, get_name, save_res
from torch.utils.data import DataLoader
from models.models import MultiGAT
from models.train_eval import update, evaluate, EarlyStopping, predict


def run(args, device):
    if args.mode == "test":
        args.debug = 0
    current_time = datetime.datetime.now()
    name = get_name(args, current_time)
    model_name = "{}.pt".format(name)

    run = wandb.init(
        # Set the project where this run will be logged
        project="gnnvuldetect",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "n_layers": args.n_layers,
            "num_head": args.num_head,
            "dropout": args.dropout,
            "hid_dim": args.hid_dim,
            "batch_size": args.batch_size,
        },
    )

    # Load Data

    console.log("Loading dataset: ", args.dataset)
    dataset = get_data(data=args.dataset, mode=args.mode)
    dataset.read_all_graphs()
    if args.debug:
        console.log("Dataset loaded, size: ", len(dataset))
        console.log("Taking a look at the first element: ", dataset[0])

    if args.mode == "train":
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        tr_idx = idx[: int(0.8 * len(dataset))]
        te_idx = idx[int(0.8 * len(dataset)) :]
        tr_data = dataset.sub_data(tr_idx)
        te_data = dataset.sub_data(te_idx)
        console.log(
            "Data split into train and test, with sizes: ", len(tr_data), len(te_data)
        )

        # Data Loader
        tr_loader = DataLoader(
            tr_data, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate
        )
        te_loader = DataLoader(
            te_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )
    else:
        te_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=custom_collate,
        )
    console.log("Data Loader created")
    # Get the first batch
    if args.debug:
        for batch in tr_loader:
            console.log("Batch size: ", len(batch))
            console.log("Batch: ", batch)
            break

    # Initialize Model
    model = MultiGAT(
        in_feats=dataset.in_dim,
        n_hidden=args.hid_dim,
        n_classes=1,
        n_layers=args.n_layers,
        num_head=args.num_head,
        dropout=args.dropout,
    )
    loss_fn = torch.nn.BCELoss()

    if args.mode == "train":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        es = EarlyStopping(patience=args.patience, verbose=False)
    else:
        model.load_state_dict(torch.load(os.path.join(args.save_path, args.model_load)))
        # model.eval()

    model = model.to(device)
    loss_fn = loss_fn.to(device)
    # optimizer = optimizer.to(device)

    if args.debug:
        console.log("Model initialized, model: ", model)
        console.log("Optimizer initialized, params: ", model.parameters())

    if args.mode == "train":
        history = {
            "train_history_loss": [],
            "val_history_loss": [],
            "val_history_acc": [],
            "val_history_auc": [],
            "val_history_f1": [],
        }

        # Train Model
        for epoch in range(args.epochs):
            tr_loss = update(
                model, optimizer, loss_fn, tr_loader, dataset.type_of_graph, device
            )
            loss, acc, auc, f1, precision = evaluate(
                model, te_loader, dataset.type_of_graph, device, loss_fn
            )
            console.log(
                f"Epoch: {epoch}, Loss: {loss}, Acc: {acc}, AUC: {auc}, F1: {f1}, Precision: {precision}"
            )
            results = {
                "Train/loss": tr_loss,
                "Val/loss": loss,
                "Val/acc": acc,
                "Val/auc": auc,
                "Val/f1": f1,
            }
            history["train_history_loss"].append(tr_loss)
            history["val_history_loss"].append(loss)
            history["val_history_acc"].append(acc)
            history["val_history_auc"].append(auc)
            history["val_history_f1"].append(f1)

            wandb.log({**results})

            es(
                epoch=epoch,
                epoch_score=acc,
                model=model,
                model_path=args.save_path + model_name,
            )

        save_res(name=name, args=args, dct=history)
    else:
        preds = predict(model, te_loader, dataset.type_of_graph, device)
        df = dataset.df.copy()
        df["pred"] = preds
        df = df.reset_index(drop=True)
        df.to_csv(os.path.join(args.res_path, f"{name}.csv"), index=False)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    seed_everything(args.seed)
    run(args, device)
