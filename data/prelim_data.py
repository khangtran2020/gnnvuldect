import os
import dgl
import json
import torch
import pickle
import numpy as np
import pandas as pd
from data.core import Data


class PrelimData(Data):

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.data_path = "Dataset/data.csv"
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
        # print(type(mask))
        mask = eval(mask)
        label = self.df.iloc[idx]["label"]
        graph_dict = self.read_graphs(path=os.path.join("Dataset", graph_name))
        return graph_dict, mask, label

    def __len__(self):
        return self.df.shape[0]

    def read_graphs(self, path: str) -> dict:
        graph_dict = {}
        num_nodes = self._get_num_nodes_from_raw(path=path)
        feat = self._read_node_features(path=path)
        assert num_nodes == feat.shape[0]
        for etype in self.type_of_graph:
            if os.path.exists(os.path.join(path, f"{etype}.pkl")):
                u, v = self._read_edge_list(path=path, etype=etype)
                graph = dgl.graph((u, v), num_nodes=num_nodes)
                graph.ndata["feat"] = feat
                graph_dict[etype] = graph
        return graph_dict

    def read_pickle(self, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

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
