import torch
from tsgcn.util import windowed_sum_pooling
from torch_geometric.nn import GCNConv, GATv2Conv, GCN2Conv
from torch.nn import Linear, BatchNorm1d
import torch.nn.functional as F


class BiGCNEncoder(torch.nn.Module):
    """
    A bidirectional GCN that takes a TreeSequenceData object as input and returns a vector of node embeddings.
    Node embeddings are updated by a GCN layer applied to edges that overlap with a window at a time, going both forwards and backwards.
    Forward and backward embeddings are summed to get the final embedding.
    TODO:
    """

    def __init__(
        self,
        num_features,
        channels,
        num_layers,
        alpha=0.1,
        theta=0.5,
        dropout=0.0,
        seed=1332339,
        **kwargs,
    ):
        super(BiGCNEncoder, self).__init__()
        # self.breaks = breaks
        # self.device = device
        self.num_features = num_features
        self.channels = channels
        self.num_layers = num_layers
        self.alpha = alpha
        self.theta = theta
        self.dropout = dropout
        self.lin = Linear(self.num_features, self.channels)
        # Convolution layers for forward and backward passes
        self.convs = torch.nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(
                GCN2Conv(
                    self.channels, self.alpha, self.theta, layer=l + 1, normalize=False
                )
            )
        # self.conv_f = GCNConv(self.out_features, self.out_features)
        # self.conv_b = GCNConv(self.out_features, self.out_features)
        self.batch_norm = BatchNorm1d(self.channels, momentum=0.1)
        # self.batch_norm_f = BatchNorm1d(self.out_features, momentum=0.1)
        # self.batch_norm_b = BatchNorm1d(self.out_features, momentum=0.1)

    def forward(self, data):
        x0 = data.x.clone()
        x = x0 = self.lin(x0).relu()
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x0, data.edge_index)
            x = self.batch_norm(x)
            x = x.relu()
        return x


class BiGCNModel(torch.nn.Module):
    def __init__(
        self,
        num_features,
        channels,
        num_layers,
        device,
        num_out_features=1,
        pooling=None,
        breaks=None,
        dropout=0.0,
        activation="identity",
        **kwargs,
    ):
        super(BiGCNModel, self).__init__()
        self.num_features = num_features
        self.channels = channels
        self.num_layers = num_layers
        self.device = device
        self.num_out_features = num_out_features
        self.breaks = breaks
        self.dropout = dropout
        self.encoder = BiGCNEncoder(
            self.num_features,
            self.channels,
            self.num_layers,
            dropout=self.dropout,
            **kwargs,
        )
        self.lin1 = Linear(channels, 16)
        self.batch_norm = BatchNorm1d(16, momentum=0.1)
        self.lin2 = Linear(16, num_out_features)
        self.pool = None
        self.pool_kwargs = {}
        if pooling == "windowed_sum":
            self.pool = windowed_sum_pooling
            self.pool_kwargs = {"breaks": kwargs.get("out_breaks")}
        self.dropout = dropout
        self.activation = torch.nn.Identity()
        if activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "relu":
            self.activation = torch.nn.ReLU()

    def forward(self, data):
        # node embeddings num_nodes x num_encoder_out_features
        x = self.encoder(data)
        # pooledI embeddings num_windows x num_encoder_out_features//2
        if self.pool is not None:
            x = self.pool(x, data, self.device, **self.pool_kwargs)
        out = self.batch_norm(self.lin1(x))
        out = F.dropout(
            self.activation(out),
            p=self.dropout / 2,
            training=self.training,
        )
        out = F.dropout(
            self.activation(self.lin2(out)), p=self.dropout / 2, training=self.training
        )
        return out
