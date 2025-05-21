import numpy as np
import random
import torch
import os
from scipy.sparse import diags

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

def get_class_num(imb_ratio, num_train, num_val,data_name,n_class,n_data):
    # Multi classification #
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

        # Ensure the imbalance ratio is maintained
        class_num_list_train[-1] = round(class_num_list_train[0] / imb_ratio)
        class_num_list_train[0] = round(class_num_list_train[-1] * imb_ratio)

        original_class_num_list_train = [class_num_list_train[inv_indices[i]] for i in range(n_class)]
        c_train_num = original_class_num_list_train
        
        c_val_num = [num_val] * n_class
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


def upsample(dataset): 
    y = torch.tensor([dataset[i].y for i in range(len(dataset))])
    classes = torch.unique(y)
    num_class_graph = [(y == i.item()).sum() for i in classes]
    max_num_class_graph = max(num_class_graph) 
    chosen = []
    for i in range(len(classes)):   
        train_idx = torch.where((y == classes[i]) == True)[0].tolist()  
        up_sample_ratio = max_num_class_graph / num_class_graph[i]  
        up_sample_num = int(num_class_graph[i] * up_sample_ratio - num_class_graph[i]) 
        if(up_sample_num <= len(train_idx)):  
            up_sample = random.sample(train_idx, up_sample_num) 
        else:
            tmp = int(up_sample_num / len(train_idx))  
            up_sample = train_idx * tmp
            tmp = up_sample_num - len(train_idx) * tmp 
            up_sample.extend(random.sample(train_idx, tmp))  
        chosen.extend(up_sample)
    chosen = torch.tensor(chosen)
    chosen = chosen.to(torch.long)  
    extend_data = dataset[chosen]
    data = list(dataset) + list(extend_data) 
    return data

def batch_to_gpu(batch, device):
    for key in batch:
        if isinstance(batch[key], list): 
            for i in range(len(batch[key])):
                batch[key][i] = batch[key][i].to(device)
        else:
            batch[key] = batch[key].to(device)
    return batch


def shuffle(dataset, c_train_num, c_val_num, y):
    classes = torch.unique(y)
    indices = []
    for i in range(len(classes)):
        index = torch.nonzero(y == classes[i]).view(-1)  
        index = index[torch.randperm(index.size(0))]
        indices.append(index)  
    train_index, val_index, test_index = [], [], []
    for i in range(len(classes)):
        train_index.append(indices[classes[i]][:c_train_num[classes[i]]])
        val_index.append(indices[classes[i]][c_train_num[classes[i]]:(c_train_num[classes[i]] + c_val_num[classes[i]])])
        test_index.append(indices[classes[i]][(c_train_num[classes[i]] + c_val_num[classes[i]]):])
    train_index = torch.cat(train_index, dim=0)
    val_index = torch.cat(val_index, dim=0)
    test_index = torch.cat(test_index, dim=0)
    train_dataset = dataset[train_index]
    val_dataset = dataset[val_index]
    test_dataset = dataset[test_index]
    return train_dataset, val_dataset, test_dataset

def load_split(load_path='', split_mode='low', load_file=None):
    if load_file:
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError("Parameter load_file is not a valid file")
    elif os.path.exists(load_path):
        load_file = os.path.join(load_path, 'split_' + split_mode + '.pt')
        print(load_file)
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
    return train_mask, val_mask, test_mask, boundary_size

def load_split1(load_path='', split_mode='high', load_file=None):
    if load_file:
        if os.path.exists(load_file):
            loaded_split = torch.load(load_file)
        else:
            raise ValueError("Parameter load_file is not a valid file")
    elif os.path.exists(load_path):
        load_file = os.path.join(load_path, 'split_topo_class_' + split_mode + '.pt')
        print(load_file)
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
    return train_mask, val_mask, test_mask, boundary_size

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

def compute_rwpe(data, pos_enc_dim):
    num_nodes = data.x.shape[0]  
    if num_nodes <= 1:
        return torch.zeros(num_nodes, pos_enc_dim)
    A = data.adj_t.to_scipy(layout="csr")
    degrees = A.sum(axis=1).A1
    degrees[degrees == 0] = 1  
    Dinv = diags(1.0 / degrees)
    M = Dinv.dot(A)
    PE = []
    M_power = M.copy()
    for _ in range(pos_enc_dim):
        PE.append(torch.from_numpy(M_power.diagonal()).float())
        M_power = M_power.dot(M)  
    PE = torch.stack(PE, dim=-1)
    return PE