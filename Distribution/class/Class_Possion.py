import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn.functional as F
from parse import parse_args
from tqdm import tqdm
import torch_geometric.transforms as T
from torch.utils.data import DataLoader
import math
from generator import *
from dataprocess import Dataset
from dataset1 import *
from model.UniImb import UniImb
import torch
from sklearn.metrics import f1_score
from collections import Counter
import time
from colorama import Fore, init
init(autoreset=True)

def runnerr(args, device):
    log_dir = f'/nips_UniImb/Distribution/class/log/{args.dataset}'
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

    for count in range(args.runs):
        random.seed(args.seed + count)
        np.random.seed(args.seed + count)
        torch.manual_seed(args.seed + count)
        torch.cuda.manual_seed_all(args.seed + count)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        train_dataset, val_dataset, test_dataset = shuffle(dataset, args.c_train_num, args.c_val_num, args.y)  
        print(len(train_dataset))
        print(len(val_dataset))
        train_dataset = upsample(train_dataset)
        val_dataset = upsample(val_dataset)
        
        print(len(train_dataset))
        print(len(val_dataset))

        train_dataset = Dataset(train_dataset, dataset, args)
        val_dataset = Dataset(val_dataset, dataset, args)
        test_dataset = Dataset(test_dataset, dataset, args)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_batch)  
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_batch)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=test_dataset.collate_batch)

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
    
            lam = 3.0  
            K = args.Prototypes
            N = args.batch_size
            k = torch.arange(1, K + 1).float().to(bias.device)
            lam_tensor = torch.tensor(lam, dtype=torch.float32, device=bias.device)  # 转成 tensor
            poisson_probs = (lam_tensor ** k) * torch.exp(-lam_tensor) / torch.tensor(
    [math.factorial(int(i)) for i in k], dtype=torch.float32, device=bias.device
)
            Z = poisson_probs.sum()
            target = (2 * N * args.TopK_2 / Z) * poisson_probs
            bias = bias - 0.001 * torch.sgn(topk - target)
                
        F1_micro[count] = test_eval['F1-micro']
        F1_macro[count] = test_eval['F1-macro']

        print(Fore.CYAN + f"Run {count + 1}: F1_macro = {F1_macro[count]}, F1_micro = {F1_micro[count]}")
        print(Fore.YELLOW + f"Best validation loss: {best_val_loss}")
        print(Fore.MAGENTA + f"Average F1_macro: {np.mean(F1_macro[:count+1])} (std = {np.std(F1_macro[:count+1])})")
        print(Fore.MAGENTA + f"Average F1_micro: {np.mean(F1_micro[:count+1])} (std = {np.std(F1_micro[:count+1])})")

        with open(log_filename, 'a') as log_file:
            log_file.write(f"\nRun {count+1}: F1_macro = {F1_macro[count]}, F1_micro = {F1_micro[count]}\n")
            log_file.write(f"Best validation loss: {best_val_loss}\n")
            log_file.write(f"Average F1_macro = {np.mean(F1_macro[:count+1])} (std = {np.std(F1_macro[:count+1])})\n")
            log_file.write(f"Average F1_micro = {np.mean(F1_micro[:count+1])} (std = {np.std(F1_micro[:count+1])})\n")
        
    return F1_micro, F1_macro

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

def eval(encoder,data_loader, args, device, epoch, bias):
    encoder.eval()
    pred, truth = [], []
    total_loss = 0
    index = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch_to_gpu(batch, device)
            data, train_idx = batch['data'], batch['train_idx']
            logits, _ = encoder(data, data.x, data.pos_enc, data.adj_t, data.ptr, index, bias,  alpha = 0)
            logits = logits[:,:args.n_class][train_idx]
            loss = F.nll_loss(logits, data.y[train_idx])
            total_loss += (loss * train_idx.shape[0]).item()
            pred.extend(logits.argmax(-1).tolist())
            truth.extend(data.y[train_idx].tolist())
    acc_c = f1_score(truth, pred, labels=np.arange(
        0, args.n_class), average=None, zero_division=0)
    acc = (np.array(pred) == np.array(truth)).sum() / len(truth)
    return {'loss': total_loss / (i + 1), 'F1-macro': np.mean(acc_c), 'F1-micro': acc}

if __name__ == '__main__':
    args = parse_args()
    print(args)
    args.path = os.getcwd()
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    path = osp.join("/nips_UniImb", 'data', 'TU') 
    dataset, args.n_feat, args.n_class = get_TUDataset(args.dataset, pre_transform=T.ToSparseTensor(), pos_enc_dim= args.pos_enc)
    labels = [data.y.item() for data in dataset]
    n_data = Counter(labels)
    n_data = np.array(list(n_data.values()))
    
    if args.dataset in ['IMDB-MULTI', "COLLAB", "ENZYMES", "Synthie"]:
        args.num_train = (int)(len(dataset) * 0.25)
        args.num_val = (int)(len(dataset) * 0.1 / args.n_class)

    args.c_train_num, args.c_val_num = get_class_num(args.imb_ratio, args.num_train, args.num_val,args.dataset,args.n_class, n_data)
    args.y = torch.tensor([data.y.item() for data in dataset])
    del labels
    del n_data
    print("n_class:", args.n_class)
    counts = torch.bincount(args.y)
    for i in range(len(counts)):
        print(f"class {i}:", counts[i])
    F1_micro, F1_macro = runnerr(args, device)  
    print('F1_macro: ', np.mean(F1_macro), np.std(F1_macro))
    print('F1_micro: ', np.mean(F1_micro), np.std(F1_micro))



