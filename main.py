import os
import torch
from config import parse_args
from data.utils import get_data, custom_collate
from utils.console import console
from torch.utils.data import DataLoader
from models.models import MultiGAT


def run(args):

    # Load Data

    console.log("Loading dataset: ", args.dataset)
    dataset = get_data(args.dataset)
    dataset.read_all_graphs()
    if args.debug:
        console.log("Dataset loaded, size: ", len(dataset))
        console.log("Taking a look at the first element: ", dataset[0])

    # Data Loader
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate
    )
    console.log("Data Loader created")
    # Get the first batch
    if args.debug:
        for batch in loader:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()
    if args.debug:
        console.log("Model initialized, model: ", model)
        console.log("Optimizer initialized, params: ", model.parameters())

    # if debug, test forward pass
    if args.debug:
        for batch in loader:
            X, Y = batch
            for i in range(len(X)):
                data, mask = X[i]
                mask_bin = torch.zeros(data["num_nodes"])
                mask_bin[mask] = 1
                mask_bin = mask_bin.view(-1, 1)
                pred = model(data, mask_bin)
                console.log("Prediction: ", pred)
                loss = loss_fn(pred, Y[i])
                console.log("Loss: ", loss)
                break
            break


if __name__ == "__main__":
    args = parse_args()
    run(args)
