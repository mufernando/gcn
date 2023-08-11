import torch
from tsgcn.util import windowed_sum_pooling
from torch_geometric.nn import (
    GCNConv,
    GATv2Conv,
    GCN2Conv,
    PNAConv,
    DenseGCNConv,
    dense_diff_pool,
)
from torch.nn import Linear, BatchNorm1d, GRUCell, Identity
import torch.nn.functional as F
from torch_geometric.nn.models import MLP


class GNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        out_dim,
        add_lin=True,
        track_running_stats=True,
    ):
        # initialize variables
        super(GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.track_running_stats = track_running_stats
        self.add_lin = add_lin
        self.lin = Identity()

        self.conv1 = DenseGCNConv(
            in_channels=self.input_dim, out_channels=self.hidden_dim
        )
        self.bn1 = BatchNorm1d(
            num_features=self.hidden_dim, track_running_stats=self.track_running_stats
        )

        self.conv2 = DenseGCNConv(
            in_channels=self.hidden_dim, out_channels=self.hidden_dim
        )
        self.bn2 = BatchNorm1d(
            num_features=self.hidden_dim, track_running_stats=self.track_running_stats
        )

        self.conv3 = DenseGCNConv(
            in_channels=self.hidden_dim, out_channels=self.out_dim
        )
        self.bn3 = BatchNorm1d(
            num_features=self.out_dim, track_running_stats=self.track_running_stats
        )

        if self.add_lin:
            self.lin = Linear(2 * self.hidden_dim + self.out_dim, self.out_dim)

    def forward(self, x, adj, mask):
        x1 = self.bn1(F.relu(self.conv1(x, adj, mask)).squeeze())
        x2 = self.bn2(F.relu(self.conv2(x1, adj, mask)).squeeze())
        x3 = self.bn3(F.relu(self.conv3(x2, adj, mask)).squeeze())
        x = torch.cat([x1, x2, x3], dim=-1)
        x = F.relu(self.lin(x))
        return x


class GNNNet(torch.nn.Module):
    def __init__(
        self, num_features, hidden_dim=64, out_GNN_dim=32, out_MLP_dim=1, dropout=0.8
    ):
        super().__init__()
        # initialize variables
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.out_GNN_dim = out_GNN_dim
        self.out_MLP_dim = out_MLP_dim
        self.dropout = dropout

        self.GNN = GNN(
            self.num_features, self.hidden_dim, self.out_GNN_dim, add_lin=False
        )
        self.MLP = MLP(
            [
                (2 * self.hidden_dim) + self.out_GNN_dim,
                2 * self.hidden_dim,
                self.hidden_dim,
                self.out_MLP_dim,
            ],
            dropout=self.dropout,
        )

    def forward(self, data):
        x, adj, mask, batch = (
            data.x,
            data.adj,
            data.mask,
            data.batch,
        )
        x = self.GNN(x, adj, mask)
        x = self.MLP(x)
        return x


class PNANet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, num_layers, out_dim, deg=None):
        # initialize variables
        super(PNANet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.deg = deg

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.grus = torch.nn.ModuleList()

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        # define layers
        for i in range(self.num_layers):
            if i == 0:
                self.convs.append(
                    GATv2Conv(
                        in_channels=self.input_dim,
                        out_channels=self.hidden_dim,
                        edge_dim=self.edge_dim,
                        heads=1,
                        add_self_loops=True,
                    )
                )
                # self.convs.append(PNAConv(in_channels=self.input_dim, out_channels=self.hidden_dim, edge_dim=self.edge_dim, aggregators=aggregators, scalers=scalers, deg=self.deg))
                self.grus.append(GRUCell(self.input_dim, self.hidden_dim))
            else:
                self.convs.append(
                    GATv2Conv(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        edge_dim=self.edge_dim,
                        heads=1,
                        add_self_loops=True,
                    )
                )
                self.grus.append(GRUCell(self.hidden_dim, self.hidden_dim))
            self.batch_norms.append(BatchNorm1d(self.hidden_dim))
        # self.readout = GATv2Conv(in_channels=self.hidden_dim, out_channels=self.out_dim, edge_dim=self.edge_dim, heads=1, add_self_loops=False)
        self.lin1 = Linear(
            self.hidden_dim, self.hidden_dim // 2
        )  # PNAConv(in_channels=self.hidden_dim, out_channels=self.out_dim, edge_dim=self.edge_dim, aggregators=aggregators, scalers=scalers, deg=deg)
        self.lin2 = Linear(self.hidden_dim // 2, self.out_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i in range(self.num_layers):
            # print(self.convs[i])
            x = self.convs[i](x, edge_index)
            # x = self.grus[i](x, y)
            # x = self.batch_norms[i](x)
            x = F.relu(x)
        # x = F.relu(self.readout(x, edge_index, edge_attr))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x


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
        # self.batch_norm = BatchNorm1d(self.channels, momentum=0.1)
        # self.batch_norm_f = BatchNorm1d(self.out_features, momentum=0.1)
        # self.batch_norm_b = BatchNorm1d(self.out_features, momentum=0.1)

    def forward(self, data):
        x0 = data.x.clone()
        x = x0 = self.lin(x0).relu()
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x0, data.edge_index)
            # x = self.batch_norm(x)
            # x = x.relu()
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
        # self.batch_norm = BatchNorm1d(16, momentum=0.1)
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
        # out = self.batch_norm(self.lin1(x))
        out = F.dropout(
            self.activation(self.lin1(x)),
            p=self.dropout,
            training=self.training,
        )
        out = F.dropout(
            self.activation(self.lin2(out)), p=self.dropout, training=self.training
        )
        return out
