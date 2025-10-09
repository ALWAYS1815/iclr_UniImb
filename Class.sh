#!/bin/bash
### MUTAG ###
python /main/main_class_imbalance.py --dataset='MUTAG' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005 --num_train=50 --num_val=50 --imb_ratio=0.1 --TopK_1=8 --TopK_2=8 --device=cuda:3 --seed=4

### PTC_MR ###
python /main/main_class_imbalance.py --dataset='PTC_MR' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005 --num_train=90 --num_val=90 --imb_ratio=0.1 --TopK_1=16 --TopK_2=8 --pos_enc=8 --device=cuda:0

### DHFR ###
python /main/main_class_imbalance.py --dataset='DHFR' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005 --num_train=120 --num_val=120 --imb_ratio=0.1 --Pertu=0 --batch_size=32 --TopK_1=32 --TopK_2=16 --seed=2  --device=cuda:7

### PROTEINS ###
python /main/main_class_imbalance.py --dataset='PROTEINS' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005 --num_train=300 --num_val=300 --imb_ratio=0.1 --batch_size=32 --memory_error=True --TopK_1=16  --TopK_2=8 --pos_enc=1  --device=cuda:4

### DD ###
python /main/main_class_imbalance.py --dataset='DD' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005 --num_train=300 --num_val=300 --imb_ratio=0.1 --batch_size=64 --TopK_1=32 --TopK_2=8 --memory_error=True --pos_enc=16 --device=cuda:1

### NCI1 ###
python /main/main_class_imbalance.py --dataset='NCI1' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005 --num_train=1000 --num_val=1000 --imb_ratio=0.1  --batch_size=64 --TopK_2=16 --TopK_1=32 --device=cuda:7

### REDDIT-B ###
python /main/main_class_imbalance.py  --dataset='REDDIT-BINARY' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005 --num_train=500 --num_val=500 --imb_ratio=0.1 --memory_error=True --batch_size=64 --TopK_2=16 --TopK_1=32   --device=cuda:0

### AIDS ###
python /main/main_class_imbalance.py --dataset='AIDS' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005 --num_train=500 --num_val=500 --imb_ratio=0.1 --batch_size=32 --TopK_1=32 --TopK_2=16  --device=cuda:0

### FRANKENSTEIN ###
python /main/main_class_imbalance.py --dataset='FRANKENSTEIN' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005 --num_train=1000 --num_val=1000 --imb_ratio=0.1  --batch_size=64 --TopK_2=16 --TopK_1=32  --pos_enc=1 --device=cuda:7

### COLLAB ###
python /main/main_class_imbalance.py --dataset='COLLAB' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005 --imb_ratio=100 --batch_size=64 --memory_error=True --Pertu=0 --TopK_2=16 --TopK_1=32   --seed=2  --device=cuda:7

### IMDB-MULTI ###  
python /main/main_class_imbalance.py --dataset='IMDB-MULTI' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005 --imb_ratio=100 --batch_size=32 --TopK_1=16 --TopK_2=8  --pos_enc=8 --Pertu=0 --device=cuda:6

### Synthie ###
python /main/main_class_imbalance.py --dataset='Synthie' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005 --imb_ratio=100 --batch_size=32 --TopK_1=16 --TopK_2=8  --device=cuda:0






