import torch
import torch.nn.functional as F
from torch.nn import Parameter

from torch_geometric.nn.inits import uniform


class DenseSAGEConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, has_weight, self_loop):
        super(DenseSAGEConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.has_weight = has_weight
        self.self_loop = self_loop
        self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))


    def forward(self, x, adj):
        N, _ = x.size()

        if self.self_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[idx, idx] = 0

        if self.has_weight == True:
            x = torch.matmul(x, self.weight) 
        
        out = torch.matmul(adj, x)
        out = out / adj.sum(dim=-1, keepdim=True).clamp(min=1)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
