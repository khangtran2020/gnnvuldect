import os
import dgl
import json
import torch
import pickle
import numpy as np
import pandas as pd
from copy import deepcopy
from data.core import Data


class PrelimData(Data):

    def __init__(self, name: str, mode: str) -> None:
        super().__init__(name=name)
        self.mode = mode
        self.name = name
        self.data_path = f"Dataset/{name}/{mode}/data.csv"
        self.type_of_graph = [
            "ARGUMENT",
            "RECEIVER",
            "CALL",
            "REACHING_DEF",
            "CDG",
            "CFG",
            "AST",
        ]
        self.df = pd.read_csv(self.data_path)

    def process(self):
        pass

    def __getitem__(self, idx):
        graph_name = self.df.iloc[idx]["graph"]
        mask = self.df.iloc[idx]["mask"]
        # print(graph_name)
        mask = torch.Tensor(eval(mask)).long()
        if self.mode == "train":
            label = self.df.iloc[idx]["label"]
        else:
            label = -1
        graph_dict = self.all_graphs[graph_name]
        graph_dict["name"] = graph_name
        return graph_dict, mask, label

    def __len__(self):
        return self.df.shape[0]

    def sub_data(self, idx):
        copy_data = deepcopy(self)
        copy_data.df = copy_data.df.iloc[idx].copy().reset_index(drop=True)
        return copy_data

    def read_graphs(self, path: str) -> dict:
        graph_dict = {}
        num_nodes = self._get_num_nodes_from_raw(path=path)
        # print(path, num_nodes)
        # self.num_nodes = num_nodes
        feat = self._read_node_features(path=path)
        self.in_dim = feat.size(dim=1)
        # self.feat_size = feat.size()
        # print(path, num_nodes, feat.size())
        assert num_nodes == feat.shape[0]
        for etype in self.type_of_graph:
            if os.path.exists(os.path.join(path, f"{etype}.pkl")):
                u, v = self._read_edge_list(path=path, etype=etype)
                graph = dgl.graph((u, v), num_nodes=num_nodes)
                graph.ndata["feat"] = feat
                graph_dict[etype] = graph
        graph_dict["num_nodes"] = num_nodes
        graph_dict["feat_size"] = feat.size()
        return graph_dict

    def read_pickle(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def read_all_graphs(self) -> None:
        all_graphs = {}
        graph_id = self.df["graph"].unique()
        for idx, graph in enumerate(graph_id):
            graph_dict = self.read_graphs(
                path=os.path.join("Dataset", self.name, self.mode, graph)
            )
            all_graphs[graph] = graph_dict
        self.all_graphs = all_graphs

    def _read_edge_list(self, path: str, etype: str):
        edge_path = os.path.join(path, f"{etype}.pkl")
        edge_list = self.read_pickle(path=edge_path)
        u = torch.Tensor([edge[0] for edge in edge_list]).long()
        v = torch.Tensor([edge[1] for edge in edge_list]).long()
        return u, v

    def _read_node_features(self, path: str):
        df = pd.read_csv(os.path.join(path, "node_feat.csv"))
        df = df.drop(["id", "CODE"], axis=1)
        # print(df.shape)
        feat_df = torch.from_numpy(df.values).float()
        # print(feat_df.size())
        feat_emb = self.read_pickle(path=os.path.join(path, "embeddings.pkl"))
        # print(feat_emb[0].shape)
        feat_emb = np.concatenate([np.expand_dims(e, 0) for e in feat_emb], axis=0)
        # print(feat_emb.shape)
        feat_emb = torch.from_numpy(feat_emb).float()
        feat = torch.cat([feat_df, feat_emb], dim=1)
        # print(feat.size())
        return feat

    def _read_node_id(self, path: str):
        node_id_path = os.path.join(path, "node_id.json")
        with open(node_id_path, "r") as f:
            return json.load(f)

    def _get_num_nodes_from_raw(self, path: str):
        return len(list(self._read_node_id(path=path).keys()))
