import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--dataset", type=str, default="MUTAG",
                        help="Choose a dataset:[MUTAG, PROTEINS, DHFR, DD, NCI1, PTC-MR, REDDIT-B]")
    
    # experiments
    parser.add_argument('--runs', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--split_mode', type=str, default="high")

    # model
    parser.add_argument('--alpha', type=float, default = 1.0) 
    parser.add_argument('--head', type=int, default = 4)
    parser.add_argument('--Prototypes', type=int, default = 8)
    parser.add_argument('--TopK_2', type=int, default = 8)
    parser.add_argument('--TopK_1', type=int, default = 8)
    parser.add_argument('--dim', type=int, default = 128)
    parser.add_argument('--n_hidden', type=int, default= 128)
    parser.add_argument('--bb', type=str, default="gin")
    parser.add_argument('--Pertu', type=int, default=1) 
    parser.add_argument('--pos_enc', type=int, default= 3)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Scheduler
    parser.add_argument('--step_size', type=int, default= 10)
    parser.add_argument('--gamma', type=int, default= 0.5)
    parser.add_argument('--scheduler', type=bool, default=False)
    parser.add_argument('--memory_error', type=bool, default=False)

    # setting
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--imb_ratio', type=float, default=0.1)
    parser.add_argument('--num_train', type=int, default=50)
    parser.add_argument('--num_val', type=int, default=50)
    parser.add_argument('--setting', type=str, default='smote')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--early_stopping', type=int, default=200)

    return parser.parse_args()
