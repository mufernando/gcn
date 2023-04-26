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
        - add a way to specify sequence breaks
        - remove manual seed
    """

    def __init__(self, in_features, out_features=4, seed=1332339):
        super(BiGCNEncoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Layers to do the Graph convolutions (return vector of node embeddings with num_hidden features)
        self.conv_f1 = GCNConv(self.in_features, self.out_features * 2)
        self.conv_f2 = GCNConv(self.out_features * 2, self.out_features)
        self.conv_b1 = GCNConv(self.in_features, self.out_features * 2)
        self.conv_b2 = GCNConv(self.out_features * 2, self.out_features)
        self.batch_norm = BatchNorm1d(self.out_features * 2, momentum=0.3)
        self.lin = Linear(self.out_features * 2, self.out_features)

    def forward(self, data):
        breaks = [0, data.sequence_length]
        x_f = data.x.clone()
        x_b = data.x.clone()
        # print(x_f.shape, x_b.shape)
        for i in range(len(breaks) - 1):
            left = breaks[i]
            right = breaks[i + 1]
            # print(left,right)
            subgraph_edge = data.get_subgraph(left, right)
            x_f = self.conv_f1(x_f, subgraph_edge)
            x_f = self.conv_f2(x_f, subgraph_edge)
        for i in range(len(breaks) - 1, 0, -1):
            left = breaks[i - 1]
            right = breaks[i]
            # print(left, right)
            subgraph_edge = data.get_subgraph(left, right)
            x_b = self.conv_b1(x_b, subgraph_edge)
            x_b = self.conv_b2(x_b, subgraph_edge)
        x = torch.concat((x_f, x_b), 1)
        # print(x_f.shape, x_b.shape)
        x = self.batch_norm(x)
        x = self.lin(x)
        return x


class BiGCNModel(torch.nn.Module):
    def __init__(
        self,
        device,
        num_encoder_in_features=None,
        num_encoder_out_features=8,
        pooling=None,
        breaks=None,
    ):
        super(BiGCNModel, self).__init__()
        self.device = device
        self.breaks = breaks
        self.encoder = BiGCNEncoder(num_encoder_in_features, num_encoder_out_features)
        self.lin1 = Linear(num_encoder_out_features, num_encoder_out_features // 2)
        self.lin2 = Linear(num_encoder_out_features // 2, 1)
        self.pool = None
        if pooling == "windowed_sum":
            self.pool = windowed_sum_pooling

    def forward(self, data):
        # node embeddings num_nodes x num_encoder_out_features
        x = self.encoder(data)
        h = self.lin1(x)
        # pooledI embeddings num_windows x num_encoder_out_features//2
        if self.pool is not None:
            h = self.pool(h, data, self.device, breaks=self.breaks)
        out = self.lin2(h)
        return out
