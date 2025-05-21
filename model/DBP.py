import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import math
from model.BalOpt import BalOpt

class Dynamic_Balnaced_Prototype(nn.Module):
    def __init__(self, dim, TopK_2, TopK_1, head=4, Prototypes=16):
        super(Dynamic_Balnaced_Prototype, self).__init__()
        self.balopt = BalOpt(dim, dim, Prototypes, head, TopK_2, TopK_1)
        self.FFN = FFN(dim, dim)

        self.norm1 = RMSNorm(dim)
        self.alpha = nn.Parameter(torch.tensor(math.log(9)))
        self.beta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

        self.gamma = nn.Parameter(torch.tensor(math.log(9)))
        self.delta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

    def forward(self, x, bias): 
        x = self.norm1(F.sigmoid(self.gamma) * x )
        a, topk_indices = self.balopt(x, bias)
        return F.sigmoid(self.alpha) * self.FFN(x) + F.sigmoid(self.beta) * (a), topk_indices

class FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3):
        super(FFN, self).__init__()
        
        self.W1 = nn.Linear(dim_in, expand_ratio * dim_in) 
        self.W2 = nn.Linear(dim_in, expand_ratio * dim_in)  
        self.W3 = nn.Linear(expand_ratio * dim_in, dim_out) 
        self.dropout =  nn.Dropout(dropout)

    def forward(self, x): 
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))
    
class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)
        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size
        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        if self.bias:
            return self.scale * x_normed + self.offset
        return self.scale * x_normed
    
