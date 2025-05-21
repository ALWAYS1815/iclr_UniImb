import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parse import parse_args
from tqdm import tqdm
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import math
import numpy as np
from generator import *
from model import *
from dataprocess import Dataset
from dataset1 import *
from torch.utils.data import Dataset as BaseDataset
from model.UniImb import UniImb
import torch 
from sklearn.metrics import f1_score, roc_auc_score
import time
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from colorama import Fore, Back, Style, init
init(autoreset=True)

def runnerr(args, device):

    log_dir = f'/nips_UniImb/Distribution/topology/log/{args.dataset}'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")  
    log_filename = f'{log_dir}/{args.dataset}_training_log_{timestamp}.txt'
    with open(log_filename, 'w') as log_file:
        log_file.write("Training hyperparameters:\n")
        for key, value in vars(args).items():
            log_file.write(f'{key}: {value}\n')
            
        log_file.write("\nTraining begins...\n")

    F1_micro = np.zeros(args.runs, dtype=float)
    F1_macro = np.zeros(args.runs, dtype=float)
    AUROC = np.zeros(args.runs, dtype=float)
    Accuracy = np.zeros(args.runs, dtype=float)
    Balanced_accuracy = np.zeros(args.runs, dtype=float)

    for count in range(args.runs):
        random.seed(args.seed + count)
        np.random.seed(args.seed + count)
        torch.manual_seed(args.seed + count)
        torch.cuda.manual_seed_all(args.seed + count)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        train_mask, val_mask, test_mask, boundary_size = load_split(load_path=osp.join(osp.dirname(osp.realpath(__file__)), 'data', 'TU', args.dataset), split_mode= args.split_mode)
        print("boundary_size:",boundary_size)
        train_dataset = dataset[train_mask]
        val_dataset = dataset[val_mask]
        test_dataset = dataset[test_mask]
        
        head_avg, tail_avg, imbalance_ratio = cal_imbalance_ratio(train_dataset, boundary_size)
        
        print("head_avg:", head_avg)
        print("tail_avg:", tail_avg)
        print("imbalance_ratio:", imbalance_ratio)
        print(len(train_dataset))
        print(len(val_dataset))
        
        train_dataset = Dataset(train_dataset, dataset, args)
        val_dataset = Dataset(val_dataset, dataset, args)
        test_dataset = Dataset(test_dataset, dataset, args)
        
        shuffle_list = []
        if args.dataset == 'PTC_MR' or args.dataset == 'FRANKENSTEIN' :
            shuffle_list = [False, False, True]
        else:
            shuffle_list = [True, False, False]
            
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=shuffle_list[0], collate_fn=train_dataset.collate_batch)  
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle_list[1], collate_fn=val_dataset.collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=shuffle_list[2], collate_fn=test_dataset.collate_batch)
            
        model = UniImb(args).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        best_val_loss = math.inf  
        val_loss_hist = []
        
        bias = torch.zeros(args.Prototypes).to(device)
        
        for epoch in tqdm(range(0, args.epochs)):
            loss, topk = train(model, train_loader, optimizer, args, device, epoch + 1, scheduler, bias)
            val_eval = eval(model, val_loader, args, device, epoch + 1, bias)
                
            if val_eval['loss'] < best_val_loss:
                best_val_loss = val_eval['loss']
                test_eval = eval(model, test_loader, args, device, epoch+1, bias)
                
            val_loss_hist.append(val_eval['loss'])
            if args.early_stopping > 0 and epoch > args.epochs // 2:
                tmp = torch.tensor(val_loss_hist[-(args.early_stopping + 1): -1])
                if val_eval['loss'] > tmp.mean().item():
                    break
                
            a = 3.0 
            K = args.Prototypes
            N = args.batch_size
            k = torch.arange(1, K + 1).float().to(bias.device)
            lam_tensor = torch.tensor(a, dtype=torch.float32, device=bias.device) 
            poisson_probs = (lam_tensor ** k) * torch.exp(-lam_tensor) / torch.tensor([math.factorial(int(i)) for i in k], dtype=torch.float32, device=bias.device)
            Z = poisson_probs.sum()
            target = (2 * N * args.TopK_2 / Z) * poisson_probs
            bias = bias - 0.001 * torch.sgn(topk - target)
            
        F1_macro[count] = test_eval['F1-macro']
        AUROC[count] = test_eval['AUROC']
        Balanced_accuracy[count] = test_eval['Balanced Accuracy']
        Accuracy[count] = test_eval['Accuracy']

        print(Fore.CYAN + f"Run {count + 1}: F1_macro = {F1_macro[count]}, AUROC = {AUROC[count]}, Balanced_accuracy= {Balanced_accuracy[count]}, Accuracy = {Accuracy[count]}, F1_micro = {F1_micro[count]}")
        print(Fore.YELLOW + f"Best validation loss: {best_val_loss}")
        print(Fore.MAGENTA + f"Average F1_macro: {np.mean(F1_macro[:count+1])} (std = {np.std(F1_macro[:count+1])})")
        print(Fore.MAGENTA + f"Average AUROC: {np.mean(AUROC[:count+1])} (std = {np.std(AUROC[:count+1])})")
        print(Fore.MAGENTA + f"Average Balanced_accuracy: {np.mean(Balanced_accuracy[:count+1])} (std = {np.std(Balanced_accuracy[:count+1])})")
        print(Fore.MAGENTA + f"Average Accuracy: {np.mean(Accuracy[:count+1])} (std = {np.std(Accuracy[:count+1])})")
        print(Fore.MAGENTA + f"Average F1_micro: {np.mean(F1_micro[:count+1])} (std = {np.std(F1_micro[:count+1])})")
        
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\nRun {count+1}: F1_macro = {F1_macro[count]}, AUROC = {AUROC[count]}, Balanced_accuracy= {Balanced_accuracy[count]}, Accuracy = {Accuracy[count]}, F1_micro = {F1_micro[count]}\n")
            log_file.write(f"Best validation loss: {best_val_loss}\n")
            log_file.write(f"Average F1_macro = {np.mean(F1_macro[:count+1])} (std = {np.std(F1_macro[:count+1])})\n")
            log_file.write(f"Average AUROC: {np.mean(AUROC[:count+1])} (std = {np.std(AUROC[:count+1])})\n")
            log_file.write(f"Average Balanced_accuracy: {np.mean(Balanced_accuracy[:count+1])} (std = {np.std(Balanced_accuracy[:count+1])})\n")
            log_file.write(f"Average Accuracy: {np.mean(Accuracy[:count+1])} (std = {np.std(Accuracy[:count+1])})\n")
            log_file.write(f"Average F1_micro = {np.mean(F1_micro[:count+1])} (std = {np.std(F1_micro[:count+1])})\n")
        
            
    return  F1_macro, AUROC, Balanced_accuracy, Accuracy

def train(encoder, data_loader, optimizer, args, device, epoch, scheduler, bias):
    encoder.train()
    total_loss = 0
    for i, batch in enumerate(data_loader):
        batch_to_gpu(batch, device)
        data, train_idx = batch['data'], batch['train_idx']
        batch_size = batch['data'].y.shape[0]
        index = torch.randperm(batch_size)
        logits, topk_indices = encoder(data, data.x, data.pos_enc, data.adj_t, data.ptr, index, bias, args.alpha)
        topk = torch.bincount(topk_indices.flatten(), minlength = args.Prototypes)  
        loss = F.nll_loss(logits[:batch_size,:args.n_class], data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.scheduler:
            scheduler.step()
        avg_train_loss = total_loss / (i + 1)
    return avg_train_loss, topk

def eval(encoder, data_loader, args, device, epoch, bias):
    encoder.eval()
    pred, truth = [], []
    probas = []
    total_loss = 0
    index = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch_to_gpu(batch, device)
            data, train_idx = batch['data'], batch['train_idx']
            logits, topk_indices = encoder(data, data.x, data.pos_enc, data.adj_t, data.ptr, index, bias, alpha=0)
            logits = logits[:, :args.n_class][train_idx]
            loss = F.nll_loss(logits, data.y[train_idx])
            total_loss += (loss * train_idx.shape[0]).item()
            probs = torch.exp(logits)  
            pred_batch = logits.argmax(dim=-1)
            pred.extend(pred_batch.tolist())
            truth_batch = data.y[train_idx].tolist()
            truth.extend(truth_batch)
            probas.extend(probs.tolist()) 
    pred_np = np.array(pred)
    truth_np = np.array(truth)
    probas_np = np.array(probas)
    acc = accuracy_score(truth_np, pred_np)
    f1_macro = f1_score(truth_np, pred_np, average='macro', zero_division=0)
    f1_micro = f1_score(truth_np, pred_np, average='micro', zero_division=0)
    balanced_acc = balanced_accuracy_score(truth_np, pred_np)
    try:
        auroc = roc_auc_score(
            np.eye(args.n_class)[truth_np],
            probas_np,
            multi_class='ovr'
        )
    except ValueError:
        auroc = -1  

    return {
        'loss': total_loss / (i + 1),
        'F1-macro': f1_macro,
        'F1-micro': f1_micro,
        'Accuracy': acc,
        'Balanced Accuracy': balanced_acc,
        'AUROC': auroc
    }

if __name__ == '__main__':
    args = parse_args()
    args.path = os.getcwd()
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    path = osp.join("/nips_UniImb", 'data', 'TU')  
    dataset, args.n_feat, args.n_class = get_TUDataset(args.dataset, pre_transform=T.ToSparseTensor(), pos_enc_dim=args.pos_enc)
    args.y = torch.tensor([data.y.item() for data in dataset])
    print("n_class:", args.n_class)
    counts = torch.bincount(args.y)
    for i in range(len(counts)):
        print(f"class {i}:", counts[i])
    F1_macro, AUROC, Balanced_accuracy, Accuracy = runnerr(args, device)  
    print('F1_macro: ', np.mean(F1_macro), np.std(F1_macro))
    print('AUROC: ', np.mean(AUROC), np.std(AUROC))
    print('Balanced_accuracy: ', np.mean(Balanced_accuracy), np.std(Balanced_accuracy))
    print('Accuracy: ', np.mean(Accuracy), np.std(Accuracy))

