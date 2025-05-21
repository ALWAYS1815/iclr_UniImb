import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.utils.dropout import dropout_edge
from torch_sparse import SparseTensor

class Graph_Pertu_Strategy(nn.Module):
    def __init__(self):
        super(Graph_Pertu_Strategy, self).__init__()
        self.pertu = MLP()
     
    def forward(self, data):
        ptr = data.ptr
        edge_index = torch.stack(data.adj_t.coo()[:2]).to(data.x.device)
        x = data.x
        avg_degree = torch.stack([data.avg_degree, data.avg_degree], dim=1)
   
        new_edge_index_list = []
        new_x_list = []
        for i in range(len(ptr) - 1):
            start, end = ptr[i], ptr[i + 1]
            graph_edge_index = edge_index[:, (edge_index[0] >= start) & (edge_index[0] < end)]
            graph_x = x[start:end]
            drop_ratio, mask_ratio = self.pertu(avg_degree)[i, 0].item(), self.pertu(avg_degree)[i, 1].item()
            graph_edge_index, _ = dropout_edge(graph_edge_index, p = drop_ratio)
            node_num = end - start
            drop_num = int(node_num * mask_ratio)
            idx_mask = torch.randperm(node_num)[:drop_num]
            new_x = graph_x.clone()
            new_x[idx_mask] = 0
            new_edge_index_list.append(graph_edge_index)
            new_x_list.append(new_x)

        edge_index_aug= torch.cat(new_edge_index_list, dim=1)
        x = torch.cat(new_x_list, dim=0).long().to(data.x.device)
        adj_t = SparseTensor(row=edge_index_aug[0], col=edge_index_aug[1], value=None, sparse_sizes=(data.x.size(0), data.x.size(0)))
        return adj_t, x

class MLP(torch.nn.Module):
    def __init__(self, n_hidden = 32):  
        super(MLP, self).__init__()
        self.lin1 = Linear(2, n_hidden)  
        self.lin2 = Linear(n_hidden, 2) 

    def forward(self, x):
        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.sigmoid(x)