from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import degree
from torch_geometric.io import read_tu_data
import torch
from typing import Optional, Callable, List
import numpy as np
import os
import shutil
import os.path as osp
from typing import Callable, List, Optional
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.io import read_tu_data
from transform import EVD_Laplacian
from generator import *



class TUDataset(InMemoryDataset):
    r"""A variety of graph kernel benchmark datasets."""
    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = False, use_edge_attr: bool = False,
                 cleaned: bool = False,
                 pos_enc_dim: int = 3):
        self.pos_enc_dim = pos_enc_dim
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
            
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]
            
        if self.name in ["REDDIT-BINARY", "COLLAB", "IMDB-BINARY", "IMDB-MULTI", "Synthie", "FRANKENSTEIN"]:
            self._add_node_features()
        self._load_eigenvalues_and_eigenvectors()
        self._load_or_compute_avg_degrees()
        self._load_or_compute_rwpe()
        self._print_statistics()

    def _add_node_features(self):
        """Add node features for specific datasets."""
        node_features = []
        x_slices = [0]
        for adj_t in self.data.adj_t:
            row, col, _ = adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
            num_nodes = adj_t.size(0)
            deg = degree(edge_index[0], num_nodes=num_nodes, dtype=torch.long)
            x = torch.ones((num_nodes, 1))  
            node_features.append(x)
            x_slices.append(x_slices[-1] + num_nodes)
        self.data.x = torch.cat(node_features, dim=0)
        self.slices['x'] = torch.tensor(x_slices, dtype=torch.long)

    def _load_eigenvalues_and_eigenvectors(self):
        """Load precomputed eigenvalues and eigenvectors."""
        processed_dir = self.processed_dir
        dataset_dir = os.path.join(processed_dir, "LapE")
        os.makedirs(dataset_dir, exist_ok=True)
        eigen_values_path = os.path.join(dataset_dir, 'eigen_values.pt')
        eigen_vectors_path = os.path.join(dataset_dir, 'eigen_vectors.pt')
        slices_path = os.path.join(dataset_dir, 'slices.pt')

        if (os.path.exists(eigen_values_path) and 
            os.path.exists(eigen_vectors_path) and 
            os.path.exists(slices_path)):
            print(f"Loading precomputed eigenvalues and eigenvectors of {self.name}...")
            self.data.eigen_values = torch.load(eigen_values_path)
            self.data.eigen_vectors = torch.load(eigen_vectors_path)
            self.slices = torch.load(slices_path)
        else:
            print(f"Starting to compute eigenvalues and eigenvectors of {self.name}...")
            eigen_values_list = []
            eigen_vectors_list = []
            eigen_values_slices = [0]
            eigen_vectors_slices = [0]
            for i in range(len(self)):
                data = self.get(i)
                row, col, _ = data.adj_t.coo()
                data.edge_index = torch.stack([row, col], dim=0)
                D, V = EVD_Laplacian(data)  
                eigen_values_list.append(D)
                eigen_vectors_list.append(V.reshape(-1))
                eigen_values_slices.append(eigen_values_slices[-1] + D.size(0))
                eigen_vectors_slices.append(eigen_vectors_slices[-1] + V.numel())
            self.data.eigen_values = torch.cat(eigen_values_list, dim=0)
            self.data.eigen_vectors = torch.cat(eigen_vectors_list, dim=0)
            self.slices['eigen_values'] = torch.tensor(eigen_values_slices, dtype=torch.long)
            self.slices['eigen_vectors'] = torch.tensor(eigen_vectors_slices, dtype=torch.long)
            torch.save(self.data.eigen_values, eigen_values_path)
            torch.save(self.data.eigen_vectors, eigen_vectors_path)
            torch.save(self.slices, slices_path)
            print(f"Eigenvalues and eigenvectors of {self.name} have been saved to {dataset_dir}")

    def _load_or_compute_avg_degrees(self):
        """Load or compute the average degree for each graph."""
        avg_degree_path = os.path.join(self.processed_dir, 'avg_degrees.pt')
        slices_path = os.path.join(self.processed_dir, 'slices.pt')
    
        if os.path.exists(avg_degree_path) and os.path.exists(slices_path):
            print(f"Loading precomputed average degrees of {self.name}...")
            self.data.avg_degree = torch.load(avg_degree_path)
            self.slices = torch.load(slices_path)
        else:
            print(f"Starting to compute average degrees of {self.name}...")
            avg_degrees_list = []
            avg_degrees_slices = [0]  
            for i in range(len(self)):
                data = self.get(i)
                row, col, _ = data.adj_t.coo()
                edge_index = torch.stack([row, col], dim=0)
                deg = degree(edge_index[0], num_nodes=data.num_nodes)
                avg_degree = deg.float().mean().item()
                avg_degrees_list.append(avg_degree)
                avg_degrees_slices.append(avg_degrees_slices[-1] + 1)  

            self.data.avg_degree = torch.tensor(avg_degrees_list)
            self.slices['avg_degree'] = torch.tensor(avg_degrees_slices, dtype=torch.long)
            
            torch.save(self.data.avg_degree, avg_degree_path)
            torch.save(self.slices, slices_path)
            print(f"Average degrees of {self.name} have been saved to {avg_degree_path}")
            
    def _load_or_compute_rwpe(self):
        pos_enc_path = os.path.join(self.processed_dir, f'rwpe_{self.pos_enc_dim}.pt')
        slices_path = os.path.join(self.processed_dir, 'slices.pt')
        if os.path.exists(pos_enc_path) and os.path.exists(slices_path):
            print(f"Loading precomputed random walk position encodings for {self.name}...")
            self.data.pos_enc = torch.load(pos_enc_path)
            self.slices = torch.load(slices_path)
        else:
            print(f"Starting to compute random walk position encodings for {self.name}...")
            pos_enc_list = []
            pos_enc_slices = [0]

            for i in range(len(self)):
                data = self.get(i)
                if hasattr(data, 'ptr'):
                    start, end = data.ptr[i].item(), data.ptr[i + 1].item()
                    subgraph = Data(
                        x=data.x[start:end],
                        edge_index=data.edge_index[:, (data.edge_index[0] >= start) & (data.edge_index[1] < end)] - start,
                        num_nodes=end - start
                    )
                else:
                    subgraph = data 
                pos_enc = compute_rwpe(subgraph, self.pos_enc_dim)
                pos_enc_list.append(pos_enc)
                pos_enc_slices.append(pos_enc_slices[-1] + pos_enc.size(0))  
            self.data.pos_enc = torch.cat(pos_enc_list, dim=0) if pos_enc_list else None
            self.slices['pos_enc'] = torch.tensor(pos_enc_slices, dtype=torch.long)
            torch.save(self.data.pos_enc, pos_enc_path)
            torch.save(self.slices, slices_path)
            print(f"Random walk position encodings for {self.name} have been saved.")
            
    def _print_statistics(self):
        """Print dataset statistics."""
        total_nodes = 0
        total_edges = 0
        degrees = []
        for i in range(len(self)):
            data = self.get(i)
            num_nodes = data.num_nodes
            num_edges = data.num_edges // 2 if not data.is_directed() else data.num_edges
            total_nodes += num_nodes
            total_edges += num_edges
            if data.edge_index is None and hasattr(data, 'adj_t'):
                row, col, _ = data.adj_t.coo()
                data.edge_index = torch.stack([row, col], dim=0)
            deg = degree(data.edge_index[0], num_nodes=num_nodes)
            degrees.append(deg)

        avg_nodes = total_nodes / len(self)
        avg_edges = total_edges / len(self)
        avg_degree = torch.cat(degrees).float().mean().item()
        node_feature_dim = self.num_node_attributes + self.num_node_labels

        print(f"Dataset Statistics for {self.name}:")
        print(f"Average #Nodes per Graph: {avg_nodes:.2f}")
        print(f"Average #Edges per Graph: {avg_edges:.2f}")
        print(f"Node Feature Dimension: {node_feature_dim}")
        print(f"Average Node Degree: {avg_degree:.2f}")
        print(self.data)
        print(torch.unique(self.data.y))
        self.data.id = torch.arange(0, self.data.y.size(0))
        self.slices['id'] = self.slices['y'].clone()
        
    @property
    def raw_dir(self) -> str:
        name = f'raw{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)  

    @property
    def processed_dir(self) -> str:
        name = f'processed{"_cleaned" if self.cleaned else ""}'
        return osp.join(self.root, self.name, name)  
    
    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels


    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.cleaned_url if self.cleaned else self.url
        folder = osp.join(self.root, self.name)
        path = download_url(f'{url}/{self.name}.zip', folder)
        extract_zip(path, folder)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        os.rename(osp.join(folder, self.name), self.raw_dir)

    def process(self):
        self.data, self.slices,_ = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def get_TUDataset(dataset, pre_transform, pos_enc_dim):
    """
    'PROTEINS', 'REDDIT-BINARY', 'MUTAG', 'PTC_MR', 'DD', 'NCI1'
    """
    path = osp.join("/nips_UniImb", 'data', 'TU')  
    dataset = TUDataset(path, name=dataset, pre_transform=pre_transform, pos_enc_dim = pos_enc_dim)
    n_feat, n_class = max(dataset.num_features, 1), dataset.num_classes
    return dataset, n_feat, n_class


