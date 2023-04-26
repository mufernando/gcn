from torch_geometric.data import Dataset, Data
import torch
import tskit
from tsgcn.util import convert_tseq
import numpy as np
import os


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
        y_func=None,
        y_name=None,
    ):
        self.seeds = seeds
        self.raw_root = raw_root
        if y_func is None:
            assert y_name is None

            def _windowed_div_from_ts(ts, num_windows=1):
                windows = np.linspace(0, ts.sequence_length, num_windows + 1)
                div = ts.diversity(windows=windows, mode="branch") / ts.diversity(
                    mode="branch"
                )
                return torch.FloatTensor(div)

            self.y_func = _windowed_div_from_ts
            self.y_name = "win-div"
        else:
            assert y_name is not None
            self.y_func = y_func
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
        self.process_y()

    def process_y(self, y_func=None, y_name=None, **kwargs):
        if y_func is None:
            assert y_name is None
            y_func = self.y_func
            y_name = self.y_name
        else:
            self.y_func = y_func
            self.y_name = y_name
        for raw_file_name, i in zip(self.raw_file_names, range(len(self.seeds))):
            ts = tskit.load(raw_file_name)
            y = y_func(ts, **kwargs)
            torch.save(y, os.path.join(self.processed_dir, f"y_{y_name}_{i}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f"tseq_{idx}.pt"))
        y = torch.load(os.path.join(self.processed_dir, f"y_{self.y_name}_{idx}.pt"))
        data.y = y
        return data
