import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_scatter import segment_csr
from transform import to_dense_list_EVD
from torch_scatter import scatter

class GIN(nn.Module):
    def __init__(self, args):
        super(GIN, self).__init__()
        self.memory_error_occurred = args.memory_error
        self.Linear_RW = Linear(args.pos_enc, args.n_feat)
        self.conv1 = GINConv(  
            Sequential(Linear(args.n_feat, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
        self.conv4 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
        self.conv5 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
         
        self.conps1 = GINConv(  
            Sequential(Linear(args.n_feat, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
        self.conps2 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
        self.conps3 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU()))
        self.conps4 = GINConv(
            Sequential(Linear(args.n_hidden, args.n_hidden), BatchNorm1d(args.n_hidden), ReLU(),
                       Linear(args.n_hidden, args.n_hidden), ReLU())) 
        self.eigVmlp = Linear(1, args.n_hidden)
        self.eigSmlp = Linear(1, args.n_hidden)
        self.MLPbackbone = Linear(args.n_hidden, args.n_hidden)
        
    def forward(self, data, x, RWPE, adj_t, batch):

        RWPE = self.Linear_RW(RWPE)
        x = x + RWPE
        x = self.conv1(x, adj_t)
        
        if not self.memory_error_occurred: 
            try:
                eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch)
                x1 = eigV_dense  
                size = scatter(torch.ones_like(x1[:, 0], dtype=torch.int64), data.batch, dim=0, reduce='add')
                mask = torch.arange(x1.size(1), device=data.x.device)[None, :] < size[:, None]
                mask = mask[data.batch]
                eigS_dense = eigS_dense.unsqueeze(-1)
                eigV_dense = eigV_dense.unsqueeze(-1)
                eigV = self.eigVmlp(eigV_dense) + self.eigVmlp(-eigV_dense)
                eigS = self.eigSmlp(eigS_dense)
                eig = eigV + eigS
                eig[~mask] = 0
                if eig[~mask].numel() > 0:
                    assert eig[~mask].max() == 0
                eigS_sign = torch.sum(eig, dim=1)
                LapPE = self.MLPbackbone(eigS_sign)
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory! Skipping LapPE computation.") 
                LapPE = torch.zeros_like(x)
                self.memory_error_occurred = True 
        else:
            LapPE = torch.zeros_like(x)
        x = x +  LapPE
        RWPE = self.conps1(RWPE, adj_t)
        x = x + RWPE 
        x = self.conv2(x, adj_t)
        RWPE = self.conps2(RWPE, adj_t)
        x = x + RWPE 
        x = self.conv3(x, adj_t)
        RWPE = self.conps3(RWPE, adj_t)
        x = x + RWPE 
        x = self.conv4(x, adj_t)
        RWPE = self.conps4(RWPE, adj_t)
        x = x + RWPE 
        x = self.conv5(x, adj_t)
        x = segment_csr(x, batch, reduce="sum")
        return x