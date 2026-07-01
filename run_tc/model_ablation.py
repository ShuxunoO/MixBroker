import torch
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch.nn import Linear

# Faithful reproduction of the three encoder backbones that the repo's model.py
# toggles by commenting/uncommenting. All are 2-layer in->32->16 with ReLU after
# layer 1 and a dot-product decoder -- identical except the conv operator.
_CONV = {'gcn': GCNConv, 'gat': GATConv, 'sage': SAGEConv}


class GNN_NET(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, backbone='sage'):
        super().__init__()
        Conv = _CONV[backbone]
        self.conv1 = Conv(in_channels, hidden_channels)
        self.conv2 = Conv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        r = (src * dst).sum(dim=-1)
        return r

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)
