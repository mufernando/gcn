import torch
import gpustat
import numpy as np
from torch_geometric.utils import degree
import torch.nn.functional as F


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


def onehot_population_encoding(tree_sequence, tips_only=True):
    nodes = tree_sequence.tables.nodes
    out = torch.zeros(
        tree_sequence.num_nodes,
        tree_sequence.num_populations,
    )
    if tips_only:
        samples = [i for i in tree_sequence.samples()]
    else:
        samples = [i for i in range(tree_sequence.num_nodes)]
    out[samples, :] = F.one_hot(
        torch.from_numpy(nodes.population[samples].astype(int)),
        tree_sequence.num_populations,
    ).type(torch.float32)
    return out


def convert_tseq(ts):
    """
    Converts a tree sequence into a tuple with three tensors: `edge_idx` (2, num_edges),
    `edge_interval` (2, num_edges), `node_features` (num_nodes, num_nodes_features)
    and the ts.sequence_length.
    The `node_features` tensor only contains the node times.
    """
    edges = ts.tables.edges
    nodes = ts.tables.nodes

    edge_span = edges.right - edges.left
    edge_length = nodes.time[edges.parent] - nodes.time[edges.child]
    span_normalize = [np.mean(edge_span), np.std(edge_span)]
    length_normalize = [np.mean(edge_length), np.std(edge_length)]

    edge_span = (edge_span - span_normalize[0]) / span_normalize[1]
    edge_span = torch.from_numpy(edge_span).type(torch.float32)
    edge_length = (edge_length - length_normalize[0]) / length_normalize[1]
    # edge_length = edge_length
    edge_length = torch.from_numpy(edge_length).type(torch.float32)
    edge_features = torch.column_stack([edge_span, edge_length])
    edge_idx = torch.LongTensor(np.row_stack((edges.parent, edges.child)))
    assert np.all(
        np.diff(np.unique(edge_idx.flatten())) == 1
    )  # there are no gaps in node ids
    node_features = onehot_population_encoding(ts, tips_only=True)
    node_features = torch.zeros(ts.num_nodes, 2, dtype=torch.float32)
    for node in ts.nodes():
        node_features[node.id, :] = torch.tensor([node.time, 0])
    norm_node_features = (
        node_features - torch.mean(node_features, axis=0)
    ) / torch.std(node_features, axis=0)
    norm_node_features = torch.eye(ts.num_nodes, 150, dtype=torch.float32)

    return edge_idx, edge_features[:, 1], norm_node_features, ts.sequence_length


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
        nodes_in_window = torch.unique(data.get_subgraph(left, right).flatten())
        pooled = torch.sum(
            x[nodes_in_window], dim=0
        )  # [0] to get the max values not indices
        x_pooled[i, :] = pooled
        # print(x_pooled,flush=True)
    return x_pooled


def get_degree_histogram(loader):
    """Returns the degree histogram to be used as input for the :obj:`deg`
    argument in :class:`PNAConv`."""
    deg_histogram = torch.zeros(1, dtype=torch.long)
    for data in loader:
        deg = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg_bincount = torch.bincount(deg, minlength=deg_histogram.numel())
        deg_histogram = deg_histogram.to(deg_bincount.device)
        if deg_bincount.numel() > deg_histogram.numel():
            deg_bincount[: deg_histogram.size(0)] += deg_histogram
            deg_histogram = deg_bincount
        else:
            assert deg_bincount.numel() == deg_histogram.numel()
            deg_histogram += deg_bincount

    return deg_histogram
