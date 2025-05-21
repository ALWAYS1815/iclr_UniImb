import os
import random
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
def get_split(dataset, head_ratio=0.2, train_ratio=0.1, val_ratio=0.1, train_head_ratio=0.1, val_head_ratio=0.1,
              from_head=False, device=None, mode='high'):
    n = len(dataset)
    num_classes = dataset.num_classes
    head_graph, tail_graph, boundary_size, K = get_head_and_tail_graph(dataset, head_ratio, from_head, device)
    for label in range(num_classes):
        if mode != 'high':
            random.shuffle(head_graph[label])
        if mode != 'low':
            random.shuffle(tail_graph[label])
    train_index, val_index, test_index = get_split_index(n, num_classes, head_graph, tail_graph, train_ratio, val_ratio,
                                                         train_head_ratio, val_head_ratio)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[train_index] = True
    val_mask[val_index] = True
    test_mask[test_index] = True

    assert not torch.logical_and(train_mask, val_mask).any().item()
    assert not torch.logical_and(test_mask, val_mask).any().item()
    assert not torch.logical_and(train_mask, test_mask).any().item()

    if device:
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

    return train_mask, val_mask, test_mask, boundary_size, K


def get_head_and_tail_graph(dataset, head_ratio=0.2, from_head=False, device=None):
    n = len(dataset)
    num_classes = dataset.num_classes
    head_graph = {label: [] for label in range(num_classes)}
    tail_graph = {label: [] for label in range(num_classes)}
    num_nodes = torch.tensor([graph.num_nodes for graph in dataset])
    if device:
        num_nodes = num_nodes.to(device)
    sorted_tensor, sorted_indices = torch.sort(num_nodes, descending=from_head)  
    boundary_index = int(n * (head_ratio if from_head else 1 - head_ratio))  
    boundary_size = sorted_tensor[boundary_index] 
    for index, size in enumerate(sorted_tensor):
        graph = dataset[sorted_indices[index]]
        if size <= boundary_size:
            tail_graph[graph.y.item()].append(sorted_indices[index].item())
        else:
            head_graph[graph.y.item()].append(sorted_indices[index].item())
    K = sum(len(graphs) for graphs in head_graph.values())  
    return head_graph, tail_graph, boundary_size, K


def get_split_index(n, num_classes, head_graph, tail_graph, train_ratio, val_ratio, train_head_ratio, val_head_ratio):
    train_size = int(n * train_ratio)
    if train_size % num_classes != 0:
        train_size = (train_size // num_classes + 1) * num_classes
    val_size = int(n * val_ratio)
    if val_size % num_classes != 0:
        val_size = (val_size // num_classes + 1) * num_classes
    test_size = n - train_size - val_size
    train_head_size = int(train_size * train_head_ratio)
    if train_head_size % num_classes != 0:
        train_head_size = (train_head_size // num_classes + 1) * num_classes
    val_head_size = int(val_size * val_head_ratio)
    if val_head_size % num_classes != 0:
        val_head_size = (val_head_size // num_classes + 1) * num_classes
    train_index = []
    val_index = []
    test_index = []
    train_head_index = train_head_size // num_classes
    val_head_index = val_head_size // num_classes
    train_tail_index = train_size // num_classes - train_head_index
    val_tail_index = val_size // num_classes - val_head_index
    for label in range(num_classes):
        assert len(head_graph[label]) > train_head_index + val_head_index
        train_index.extend(head_graph[label][:train_head_index])
        val_index.extend(head_graph[label][train_head_index: train_head_index + val_head_index])
        test_index.extend(head_graph[label][train_head_index + val_head_index:])
        assert len(tail_graph[label]) > train_tail_index + val_tail_index
        train_index.extend(tail_graph[label][:train_tail_index])
        val_index.extend(tail_graph[label][train_tail_index: train_tail_index + val_tail_index])
        test_index.extend(tail_graph[label][train_tail_index + val_tail_index:])

    assert len(train_index) == train_size
    assert len(val_index) == val_size
    assert len(test_index) == test_size

    return train_index, val_index, test_index


def cal_imbalance_ratio(dataset, boundary_size):
    head_size = []
    tail_size = []
    for g in dataset:
        if g.num_nodes <= boundary_size:
            tail_size.append(g.num_nodes)
        else:
            head_size.append(g.num_nodes)

    head_avg = round(np.mean(head_size), 4)
    tail_avg = round(np.mean(tail_size), 4)
    imbalance_ratio = round(head_avg / tail_avg, 4)
    return head_avg, tail_avg, imbalance_ratio


def save_split(train_mask, val_mask, test_mask, boundary_size, split_mode, save_path='', save_file=None):
    save_tensor = {
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'boundary_size': boundary_size
    }
    if save_file:
        if os.path.exists(os.path.dirname(save_file)):
            torch.save(save_tensor, save_file)
        else:
            raise ValueError("Parameter save_file is not a valid path")
    elif os.path.exists(save_path):
        save_file = os.path.join(save_path, 'split_' + split_mode + '.pt')
        torch.save(save_tensor, save_file)
    elif os.path.exists(os.path.dirname(save_path)):
        os.mkdir(save_path)
        save_file = os.path.join(save_path, 'split_' + split_mode + '.pt')
        torch.save(save_tensor, save_file)
    else:
        raise ValueError("Fail to save split file, please check parameter save_path or save_file")


def load_split(load_path='', split_mode='low', load_file=None, device=None):
    if load_file:
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError("Parameter load_file is not a valid file")
    elif os.path.exists(load_path):
        load_file = os.path.join(load_path, 'split_' + split_mode + '.pt')
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError("Cannot find split.pt in load_path")
    else:
        raise ValueError("Fail to load split file, please check parameter load_path or load_file")
    train_mask = loaded_split['train_mask']
    val_mask = loaded_split['val_mask']
    test_mask = loaded_split['test_mask']
    boundary_size = loaded_split['boundary_size']

    if device:
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)

    return train_mask, val_mask, test_mask, boundary_size


if __name__ == '__main__':
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'TU')
    data_list = [ "PTC_MR",'FRANKENSTEIN', 'DD', 'PROTEINS', 'IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB','NCI1','IMDB-MULTI']
    for data_name in data_list:
        dataset = TUDataset(path, name=data_name)
        for split_mode in ['low', 'medium', 'high']:
            train_mask, val_mask, test_mask, boundary_size, K = get_split(dataset, from_head=True, device=device)
            save_split(train_mask, val_mask, test_mask, boundary_size, split_mode,
                        save_path=os.path.join('/nips_UniImb/data/TU', data_name))
            print(data_name, split_mode, 'Finished')
