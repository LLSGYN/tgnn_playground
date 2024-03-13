import dgl
import torch as th
import torch.nn as nn

from modules import GraphSAT, TemporalSAT

class DySAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_heads,
                 n_layers,
                 n_timestamps,
                 activation) -> None:
        super(DySAT, self).__init__()
        self.n_timestamps = n_timestamps
        self.graph_aggr = nn.ModuleList()
        self.temporal_aggr = nn.ModuleList()
    
    def forward(self, g_list):
        xs = []
        # possible to parrelize here
        for g in g_list:
            xs.append(self.graph_aggr(g))
        
        node_emb = th.vstack(xs) # [T, batch_size, F]