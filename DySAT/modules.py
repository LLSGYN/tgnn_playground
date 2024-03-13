import dgl
import math
import torch as th
import torch.nn as nn

class TemporalSAT(nn.Module):
    def __init__(self, in_dim, out_dim, n_timestamps, n_heads=1):
        super(TemporalSAT, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_timestamps = n_timestamps
        self.n_heads = n_heads
        self.w_q = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.w_k = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.w_v = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        tgt_mask = th.zeros(self.n_timestamps, self.n_timestamps)
        nopeak_mask = th.tril(th.ones(self.n_timestamps, self.n_timestamps), 
                              diagonal=-1).bool()
        self.mask = tgt_mask.masked_fill(nopeak_mask, -1e9)
    
    def forward(self, x):
        # x: batch_size * T * D
        wq_x = self.w_q(x) # batch_size * T * F
        wk_x = self.w_k(x) # batch_size * T * F
        e = th.matmul(wq_x, th.transpose(wk_x)) / math.sqrt(x.shape[-1]) + self.mask
