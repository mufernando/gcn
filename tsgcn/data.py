from torch_geometric.data import Dataset, Data
import torch
import tskit
from tsgcn.util import convert_tseq
import numpy as np
import os
from functools import partial
import multiprocessing.pool as mp


class TreeSequenceData(Data):
    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        edge_interval=None,
        sequence_length=None,
    ):
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.edge_interval = edge_interval
        self.sequence_length = sequence_length

    def get_subgraph(self, left, right):
        # selecting edges that overlap with the interval [left, right)
        overlap = torch.logical_and(
            self.edge_interval[0, :] < right, self.edge_interval[1, :] >= left
        )
        return self.edge_index[:, overlap]


class TreeSequencesDataset(Dataset):
    def __init__(
        self,
        root,
        raw_root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        seeds=None,
        y_name=None,
    ):
        self.raw_root = raw_root
        self.seeds = seeds
        self.y_name = y_name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [f"{self.raw_root}sim_{s}.trees" for s in self.seeds]

    @property
    def processed_file_names(self):
        return [f"tseq_{i}.pt" for i in range(len(self.seeds))]

    def download(self):
        pass

    def process(self):
        for raw_file_name, i in zip(self.raw_file_names, range(len(self.seeds))):
            ts = tskit.load(raw_file_name)
            edge_idx, edge_int, node_features, seq_len = convert_tseq(ts)
            data = TreeSequenceData(
                x=node_features,
                edge_index=edge_idx,
                edge_interval=edge_int,
                sequence_length=seq_len,
            )
            torch.save(data, os.path.join(self.processed_dir, f"tseq_{i}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"tseq_{idx}.pt"))
        y = torch.load(os.path.join(self.processed_dir, f"y_{self.y_name}_{idx}.pt"))
        data.y = y
        return data


def windowed_div_from_ts(ts, num_windows=100):
    windows = np.linspace(0, ts.sequence_length, num_windows + 1)
    div = ts.diversity(windows=windows, mode="branch") / ts.diversity(mode="branch")
    return torch.FloatTensor(div)


def get_node_features(ts):
    num_child = np.zeros(ts.num_nodes)
    num_samples = np.zeros(ts.num_nodes)
    trees = np.zeros(ts.num_nodes)
    for tree in ts.trees():
        for node in tree.nodes():
            num_child[node] += tree.num_children(node)
            num_samples[node] += tree.num_samples(node)
            trees[node] += 1
    num_child = num_child / trees
    num_samples = num_samples / trees
    norm_node_features = (
        num_samples  # num_samples  # np.column_stack([num_child, num_samples])
    )
    norm_node_features = (
        norm_node_features - np.mean(norm_node_features, axis=0)
    ) / np.std(norm_node_features, axis=0)
    return torch.FloatTensor(norm_node_features)


def _compute_y(i, dt, y_func, y_name, **kwargs):
    seed = dt.seeds[i]
    raw_file_name = f"{dt.raw_root}sim_{seed}.trees"
    y_fname = os.path.join(dt.processed_dir, f"y_{y_name}_{i}.pt")
    if not os.path.exists(y_fname):
        ts = tskit.load(raw_file_name)
        y = y_func(ts, **kwargs)
        torch.save(y, y_fname)
    return y_fname


def compute_ys(dt, y_func, y_name, n_workers=6, **kwargs):
    compute = partial(
        _compute_y,
        dt=dt,
        y_func=y_func,
        y_name=y_name,
        **kwargs,
    )
    with mp.ThreadPool(n_workers) as pool:
        ys = pool.map(compute, range(len(dt.seeds)))
    assert len(list(ys)) == len(dt.seeds)
