import os
import torch
from config import parse_args
from data.utils import get_data, custom_collate
from utils.console import console
from torch.utils.data import DataLoader
from models.models import MultiGAT
from models.train_eval import update, evaluate


def run(args, device):

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

    model = model.to(device)
    loss_fn = loss_fn.to(device)
    # optimizer = optimizer.to(device)

    if args.debug:
        console.log("Model initialized, model: ", model)
        console.log("Optimizer initialized, params: ", model.parameters())

    # Train Model
    for epoch in range(args.epochs):
        _ = update(model, optimizer, loss_fn, loader, dataset.type_of_graph, device)
        loss, acc, auc, f1, precision = evaluate(
            model, loader, dataset.type_of_graph, device, loss_fn
        )
        console.log(
            f"Epoch: {epoch}, Loss: {loss}, Acc: {acc}, AUC: {auc}, F1: {f1}, Precision: {precision}"
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()
    run(args, device)
