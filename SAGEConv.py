import torch
import torch.nn.functional as F
from torch.nn import Parameter
# from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.utils import remove_self_loops, add_self_loops

# from ..inits import uniform


class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(SAGEConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
    #     self.normalize = normalize
    #     self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

    #     if bias:
    #         self.bias = Parameter(torch.Tensor(out_channels))
    #     else:
    #         self.register_parameter('bias', None)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     # size = self.weight.size(0)
    #     uniform(self.in_channels, self.weight)
    #     uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, size=None):
        # x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        # if self.bias is not None:
        #     aggr_out = aggr_out + self.bias

        # if self.normalize:
        #     aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
