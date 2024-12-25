import dgl
import dgl.nn.pytorch as dglnn
import torch.nn
from torch import nn


class MultiGAT(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_hidden: int,
        n_classes: int,
        n_layers: int,
        num_head: int = 8,
        dropout: int = 0.2,
    ) -> None:
        super().__init__()
        self.classification_layer = torch.nn.Linear(
            in_features=n_hidden, out_features=n_classes
        )
        self.last_activation = (
            torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
        )
        print(f"Using activation for last layer {self.last_activation}")
        self.type_of_graph = [
            "ARGUMENT",
            "RECEIVER",
            "CALL",
            "REACHING_DEF",
            "CDG",
            "CFG",
            "AST",
        ]
        self.model_argument = GAT(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            num_head=num_head,
            dropout=dropout,
        )
        self.model_receiver = GAT(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            num_head=num_head,
            dropout=dropout,
        )
        self.model_call = GAT(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            num_head=num_head,
            dropout=dropout,
        )
        self.model_reaching_def = GAT(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            num_head=num_head,
            dropout=dropout,
        )
        self.model_cdg = GAT(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            num_head=num_head,
            dropout=dropout,
        )
        self.model_cfg = GAT(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            num_head=num_head,
            dropout=dropout,
        )
        self.model_ast = GAT(
            in_feats=in_feats,
            n_hidden=n_hidden,
            n_classes=n_classes,
            n_layers=n_layers,
            num_head=num_head,
            dropout=dropout,
        )

    def forward(self, graph_dict: dict, mask: torch.Tensor):
        i = 0
        for key in self.type_of_graph:
            if key in graph_dict:
                if key == "ARGUMENT":
                    if i == 0:
                        h_overall = self.model_argument.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                    else:
                        h = self.model_argument.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                        h_overall = h_overall + h
                    i += 1
                elif key == "RECEIVER":
                    if i == 0:
                        h_overall = self.model_receiver.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                    else:
                        h = self.model_receiver.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                        h_overall = h_overall + h
                    i += 1
                elif key == "CALL":
                    if i == 0:
                        h_overall = self.model_call.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                    else:
                        h = self.model_call.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                        h_overall = h_overall + h
                    i += 1
                elif key == "REACHING_DEF":
                    if i == 0:
                        h_overall = self.model_reaching_def.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                    else:
                        h = self.model_reaching_def.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                        h_overall = h_overall + h
                    i += 1
                elif key == "CDG":
                    if i == 0:
                        h_overall = self.model_cdg.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                    else:
                        h = self.model_cdg.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                        h_overall = h_overall + h
                    i += 1
                elif key == "CFG":
                    if i == 0:
                        h_overall = self.model_cfg.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                    else:
                        h = self.model_cfg.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                        h_overall = h_overall + h
                    i += 1
                elif key == "AST":
                    if i == 0:
                        h_overall = self.model_ast.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                    else:
                        h = self.model_ast.graph_forward(
                            g=graph_dict[key],
                            x=graph_dict[key].ndata["feat"],
                            mask=mask,
                        )
                        h_overall = h_overall + h
                    i += 1
        h_overall = h_overall / i
        h_overall = self.classification_layer(h_overall)
        h_overall = self.last_activation(h_overall)
        return h_overall


class GAT(nn.Module):

    def __init__(
        self,
        in_feats: int,
        n_hidden: int,
        n_classes: int,
        n_layers: int,
        num_head: int = 8,
        dropout: int = 0.2,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            dglnn.GATConv(
                in_feats, n_hidden, num_heads=num_head, allow_zero_in_degree=True
            )
        )
        for i in range(0, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    n_hidden * num_head,
                    n_hidden,
                    num_heads=num_head,
                    allow_zero_in_degree=True,
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.nn.SELU()

    def block_forward(self, blocks: list, x: torch.Tensor, mask: torch.Tensor):
        h = x
        for i in range(0, self.n_layers):
            h_dst = h[: blocks[i].num_dst_nodes()]
            h = self.layers[i](blocks[i], (h, h_dst))
            h = self.activation(h)
            h = h.flatten(1)
        h = h.mean(1)
        h = h * mask
        h = h.mean(0)
        return h

    def graph_forward(self, g: dgl.DGLGraph, x: torch.Tensor, mask: torch.Tensor):

        h = x
        for i in range(0, self.n_layers):
            h = self.layers[i](g, h)
            h = self.activation(h)
            h = h.flatten(1)
        print("Size of h after hidden layers:", h.size())
        h = h.mean(1)
        h = h * mask
        h = h.mean(0)
        return h
