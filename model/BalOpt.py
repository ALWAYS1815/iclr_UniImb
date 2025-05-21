import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import math

class BalOpt(nn.Module):
    def __init__(self, dim, dim_attn, Prototypes, head, TopK_2, TopK_1):
        super(BalOpt, self).__init__()
        if Prototypes == 0:
            raise 
        else:
            print('The number of Prototypes is', Prototypes)

        if TopK_1 == 0:
            raise 
        else:
            print('The number of TopK Graphs is', TopK_1)    

        if TopK_2 == 0:
            raise 
        else:
            print('The number of TopK Prototypes is', TopK_2)

        self.head_dim = dim_attn // head  
        self.query = nn.Linear(dim, dim_attn)
        self.key = nn.Parameter(torch.randn((Prototypes, head, self.head_dim)))
        self.value = nn.Linear(dim, dim)
        
        self.emb = nn.Parameter(torch.full((head, Prototypes), 0).unsqueeze(0).float()) 
        self.alpha = nn.Parameter(torch.tensor([math.log(9) for _ in range(head)]).unsqueeze(-1))
        self.beta = nn.Parameter(torch.tensor([math.log(0.01) for _ in range(head)]).unsqueeze(-1))

        self.Prototypes = Prototypes
        self.head = head
        self.TopK_2 = TopK_2
        self.TopK_1 = TopK_1

    def forward(self, x, bias):
        n, f = x.shape
        q = self.query(x).reshape(n, self.head, self.head_dim) 
        attn = torch.einsum('nhd, rhd -> nhr', q, self.key) / (self.head_dim**0.5) 
        _, topk_indices = torch.topk(F.sigmoid(attn) + bias, self.TopK_2, dim=-1)            
        mask = torch.zeros_like(attn, dtype=torch.bool).scatter_(-1, topk_indices, 1) 
        _, topk_indices2 = torch.topk(attn, self.TopK_1, dim = 0)
        mask2 = torch.zeros_like(attn, dtype=torch.bool).scatter_(0, topk_indices2, 1)
        x = self.value(x).reshape(n, self.head, self.head_dim)
        v = torch.einsum('nhr, nhd -> rhd', F.sigmoid(attn + mask2.log() + self.emb), x)   
        v = torch.einsum('nhr, rhd -> nhd', F.softmax(attn + mask.log(), dim=-1), v)
        v = F.sigmoid(self.alpha)*x + F.sigmoid(self.beta) * v 
        return v.reshape(n, f), topk_indices
  