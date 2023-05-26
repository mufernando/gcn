import torch
import gpustat
import numpy as np


def get_idle_gpu():
    """
    Utility function which uses the gpustat module to select the
    least busy GPU that is available and then sets the
    CUDA_VISIBLE_DEVICES environment variable so that
    only that GPU is used
    """
    try:
        stats = gpustat.GPUStatCollection.new_query()
        ids = map(lambda gpu: int(gpu.entry["index"]), stats)
        ratios = map(
            lambda gpu: float(gpu.entry["memory.used"])
            / float(gpu.entry["memory.total"]),
            stats,
        )
        bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
        return bestGPU

    except Exception:
        pass


def convert_tseq(ts):
    """
    Converts a tree sequence into a tuple with three tensors: `edge_idx` (2, num_edges),
    `edge_interval` (2, num_edges), `node_features` (num_nodes, num_nodes_features)
    and the ts.sequence_length.
    The `node_features` tensor only contains the node times.
    """
    edges = ts.tables.edges
    edge_idx = torch.LongTensor(np.row_stack((edges.parent, edges.child)))
    edge_interval = torch.FloatTensor(np.row_stack((edges.left, edges.right)))
    node_features = []
    assert np.all(
        np.diff(np.unique(edge_idx.flatten())) == 1
    )  # there are no gaps in node ids
    for node in ts.nodes():
        node_features.append([node.time, 0])
    node_features = torch.FloatTensor(node_features)
    return edge_idx, edge_interval, node_features, ts.sequence_length


def windowed_sum_pooling(x, data, device, breaks=None):
    """
    Pooling function that pools the embeddings of nodes in a window by summing them.
    Returns a tensor of shape (num_windows, num_encoder_out_features)
    """
    if breaks is None:
        breaks = [0, data.sequence_length]
    x_pooled = torch.zeros(len(breaks) - 1, x.shape[1]).to(device)
    for i in range(len(breaks) - 1):
        left = breaks[i]
        right = breaks[i + 1]
        nodes_in_window = data.get_subgraph(left, right).flatten()
        pooled = torch.sum(
            x[nodes_in_window], dim=0
        )  # [0] to get the max values not indices
        x_pooled[i, :] = pooled
        # print(x_pooled,flush=True)
    return x_pooled
