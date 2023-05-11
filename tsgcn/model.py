import torch
from tsgcn.util import windowed_sum_pooling
from torch_geometric.nn import GCNConv
from torch.nn import Linear, BatchNorm1d


class BiGCNEncoder(torch.nn.Module):
    """
    A bidirectional GCN that takes a TreeSequenceData object as input and returns a vector of node embeddings.
    Node embeddings are updated by a GCN layer applied to edges that overlap with a window at a time, going both forwards and backwards.
    Forward and backward embeddings are summed to get the final embedding.
    TODO:
    """

    def __init__(self, breaks, device, in_features, out_features=4, seed=1332339):
        super(BiGCNEncoder, self).__init__()
        self.breaks = breaks
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        # Blowing up the input features dimension to self.out_features
        self.lin1 = Linear(self.in_features, self.out_features)
        # Convolution layers for forward and backward passes
        self.conv = GCNConv(self.out_features, self.out_features)
        self.conv_f = GCNConv(self.out_features, self.out_features)
        self.conv_b = GCNConv(self.out_features, self.out_features)
        self.batch_norm = BatchNorm1d(self.out_features, momentum=0.1)
        self.batch_norm_f = BatchNorm1d(self.out_features, momentum=0.1)
        self.batch_norm_b = BatchNorm1d(self.out_features, momentum=0.1)
        # Linear layer to combine the forward and backward embeddings
        self.lin2 = Linear(self.out_features * 3, self.out_features)

    def forward(self, data):
        x_a = data.x.clone()
        x_f = data.x.clone()
        x_b = data.x.clone()
        x_f = self.lin1(x_f)
        x_b = self.lin1(x_b)
        x_a = self.lin1(x_a)
        for i in range(len(self.breaks) - 1):
            left = self.breaks[i]
            right = self.breaks[i + 1]
            subgraph_edge = data.get_subgraph(left, right)
            x_f = self.conv_f(x_f, subgraph_edge)
            x_f = self.batch_norm_f(x_f)
        for i in range(len(self.breaks) - 1, 0, -1):
            left = self.breaks[i - 1]
            right = self.breaks[i]
            # print(left, right)
            subgraph_edge = data.get_subgraph(left, right)
            x_b = self.conv_b(x_b, subgraph_edge)
            x_b = self.batch_norm_b(x_b)
        x_a = self.conv(x_a, data.edge_index)
        x_a = self.batch_norm(x_a)
        x = torch.concat((x_a, x_f, x_b), 1)
        x = self.lin2(x)
        return x


class BiGCNModel(torch.nn.Module):
    def __init__(
        self,
        device,
        num_encoder_in_features=None,
        num_encoder_out_features=8,
        pooling=None,
        breaks=None,
        **kwargs
    ):
        super(BiGCNModel, self).__init__()
        self.device = device
        self.breaks = breaks
        self.encoder = BiGCNEncoder(
            self.breaks, self.device, num_encoder_in_features, num_encoder_out_features
        )
        self.lin1 = Linear(num_encoder_out_features, 1)
        # self.lin2 = Linear(8, 1)
        self.pool = None
        self.pool_kwargs = {}
        if pooling == "windowed_sum":
            self.pool = windowed_sum_pooling
            self.pool_kwargs = {"breaks": kwargs.get("out_breaks")}

    def forward(self, data):
        # node embeddings num_nodes x num_encoder_out_features
        x = self.encoder(data)
        # pooledI embeddings num_windows x num_encoder_out_features//2
        if self.pool is not None:
            x = self.pool(x, data, self.device, **self.pool_kwargs)
        out = self.lin1(x)
        # out = self.lin2(out)
        return out
