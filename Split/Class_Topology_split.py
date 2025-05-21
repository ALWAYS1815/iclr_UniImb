import os
import random
import torch
import numpy as np
from torch_geometric.datasets import TUDataset
from collections import Counter

def get_split(dataset, c_train_num, c_val_num, head_ratio=0.2, train_head_ratio=0.1, val_head_ratio=0.1,
              from_head=False, device=None, mode='high'):
    n = len(dataset)
    num_classes = dataset.num_classes
    head_graph, tail_graph, boundary_size, K = get_head_and_tail_graph(dataset, head_ratio, from_head, device)
    for label in range(num_classes):
        if mode != 'high':
            random.shuffle(head_graph[label])
        if mode != 'low':
            random.shuffle(tail_graph[label])
    train_index, val_index, test_index = get_split_index(head_graph, tail_graph, c_train_num, c_val_num, train_head_ratio, val_head_ratio)

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

def get_split_index(
    head_graph, tail_graph,
    c_train_num, c_val_num,  
    train_head_ratio, val_head_ratio
):
    num_classes = len(c_train_num)
    train_index, val_index, test_index = [], [], []

    for label in range(num_classes):
        heads = head_graph[label]
        tails = tail_graph[label]
        random.shuffle(heads)
        random.shuffle(tails)

        train_num = c_train_num[label]
        val_num = c_val_num[label]

        train_head_num = int(train_num * train_head_ratio)
        train_tail_num = train_num - train_head_num

        val_head_num = int(val_num * val_head_ratio)
        val_tail_num = val_num - val_head_num

        assert len(heads) >= train_head_num + val_head_num, f"Head graphs for class {label} not enough."
        assert len(tails) >= train_tail_num + val_tail_num, f"Tail graphs for class {label} not enough."

        train_index.extend(heads[:train_head_num])
        train_index.extend(tails[:train_tail_num])

        val_index.extend(heads[train_head_num:train_head_num+val_head_num])
        val_index.extend(tails[train_tail_num:train_tail_num+val_tail_num])
        
        test_index.extend(heads[train_head_num+val_head_num:])
        test_index.extend(tails[train_tail_num+val_tail_num:])

    return train_index, val_index, test_index

def get_class_num(imb_ratio, num_train, num_val,data_name,n_class,n_data):
    # Multi classification ##
    if data_name in ['IMDB-MULTI', "COLLAB", "ENZYMES", "Synthie"]:
        n_data_tensor = torch.tensor(n_data)
        sorted_n_data, indices = torch.sort(n_data_tensor, descending=True)  
        inv_indices = torch.argsort(indices)
        max_class_num_train = num_train / (1 + (n_class - 2) + (1 / imb_ratio))
        min_class_num_train = max_class_num_train / imb_ratio

        mu_train = (min_class_num_train / max_class_num_train) ** (1 / (n_class - 1))
        class_num_list_train = [round(max_class_num_train)] + [round(max_class_num_train * (mu_train ** i)) for i in range(1, n_class - 1)] + [round(min_class_num_train)]

        current_total_train = sum(class_num_list_train)
        scale_factor_train = num_train / current_total_train
        class_num_list_train = [round(num * scale_factor_train) for num in class_num_list_train]
        class_num_list_train[-1] = round(class_num_list_train[0] / imb_ratio)
        class_num_list_train[0] = round(class_num_list_train[-1] * imb_ratio)

        original_class_num_list_train = [class_num_list_train[inv_indices[i]] for i in range(n_class)]
        c_train_num = original_class_num_list_train
        c_val_num = c_train_num
    else:
        ## Binary classification ##
        if data_name in ['PTC_MR', "PROTEINS"]:
            c_train_num = [num_train - int(imb_ratio * num_train), int(imb_ratio * num_train)]
            c_val_num = [num_val - int(imb_ratio * num_val), int(imb_ratio * num_val)]
        else:
            c_train_num = [int(imb_ratio * num_train), num_train - int(imb_ratio * num_train)]
            c_val_num = [int(imb_ratio * num_val), num_val - int(imb_ratio * num_val)]
    print("Train samples per class:", c_train_num)
    print("Validation samples per class:", c_val_num)

    return c_train_num, c_val_num


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
        save_file = os.path.join(save_path, 'split_topo_class_' + split_mode + '.pt')
        torch.save(save_tensor, save_file)
    elif os.path.exists(os.path.dirname(save_path)):
        os.mkdir(save_path)
        save_file = os.path.join(save_path, 'split_topo_class_' + split_mode + '.pt')
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
        load_file = os.path.join(load_path, 'split_topo_class_' + split_mode + '.pt')
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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'dataset', 'TU')
    data_list = ['PROTEINS', 'DD', 'NCI1', 'REDDIT-BINARY']
    imb_ratio = 0.1
    split_mode = "high"
    for data_name in data_list:
        
        dataset = TUDataset(path, name=data_name)
        num_train = (int)(len(dataset) * 0.1)
        num_val = (int)(len(dataset) * 0.1)
        n_feat, n_class = max(dataset.num_features, 1), dataset.num_classes
        labels = [data.y.item() for data in dataset]
        n_data = Counter(labels)
        n_data = np.array(list(n_data.values()))
        c_train_num, c_val_num = get_class_num(imb_ratio, num_train, num_val, data_name, n_class, n_data)
        train_mask, val_mask, test_mask, boundary_size, K = get_split(dataset, c_train_num, c_val_num, from_head=True, device=device, mode =split_mode)
        save_split(train_mask, val_mask, test_mask, boundary_size, split_mode,
                       save_path=os.path.join('/nips_UniImb/data/TU', data_name))
        print(data_name, split_mode, 'Finished')
