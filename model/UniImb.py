import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from model.Pertu import Graph_Pertu_Strategy
from model.DBP import *
from backbone.GIN import GIN 
from backbone.GCN import GCN 
from backbone.GraphSAGE import SAGE

class UniImb(nn.Module):
    def __init__(self, args):  
        super(UniImb, self).__init__()
        
        self.encoder = FFN(args.dim, args.dim)  
        self.decoder = FFN(args.dim, args.n_class)   
        self.norm = RMSNorm(args.dim)
        self.DBP = Dynamic_Balnaced_Prototype(args.dim, args.TopK_2, args.TopK_1, args.head, args.Prototypes)
        self.alpha = nn.Parameter(torch.tensor(0.))
        self.beta = nn.Parameter(torch.tensor(0.))
        
        if args.bb == 'gin':
            self.embedding = GIN(args)
        elif args.bb == 'gcn':
            self.embedding = GCN(args)
        else:
            self.embedding = SAGE(args)
            
        self.aug = Graph_Pertu_Strategy()
        self.TopK_1 = args.TopK_1
        self.TopK_2 = args.TopK_2
        self.Pertu = args.Pertu

    def forward(self, data, x, RWPE, adj_t, batch, index, bias, alpha): 

        if self.Pertu:  
            adj_t, x = self.aug(data)
            
        x = self.embedding(data, x, RWPE, adj_t, batch)
        
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)  
            mixed_x = lam * x + (1 - lam) * x[index, :]  
            x = torch.cat((x, mixed_x), dim = 0)
        else:
            lam = 0
            
        x = self.encoder(x)
        c, topk_indices = self.DBP(x, bias)
        x = F.sigmoid(self.alpha) * c
        x = self.decoder(x)
        
        return F.log_softmax(x, dim=1), topk_indices
    





        






