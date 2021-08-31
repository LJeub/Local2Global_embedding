import torch.nn as nn
import torch_geometric.nn as tg_nn


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, act, bias=True):
        super(GCN, self).__init__()
        self.conv = tg_nn.GCNConv(in_channels=in_ft, out_channels=out_ft, bias=bias)
        self.act = nn.PReLU() if act == 'prelu' else act
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self.act, 'reset_parameters'):
            self.act.reset_parameters()
        elif isinstance(self.act, nn.PReLU):
            self.act.weight.data.fill_(0.25)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj):
        out = self.conv(seq, adj)
        
        return self.act(out)

