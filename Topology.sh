#!/bin/bash
python /main/main_topology_imbalance.py--dataset='PTC_MR' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005  --TopK_1=16 --pos_enc=8  --split_mode=high --bb=gin --device=cuda:6

python /main/main_topology_imbalance.py--dataset='FRANKENSTEIN' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005   --batch_size=64 --TopK_2=16 --TopK_1=32  --pos_enc=16 --split_mode=high --memory_error=True --bb=gin --device=cuda:1  

python /main/main_topology_imbalance.py --dataset='PROTEINS' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005   --batch_size=32 --TopK_1=32  --memory_error=True  --pos_enc=1 --split_mode=high  --bb=gin  --device=cuda:6

python /main/main_topology_imbalance.py--dataset='DD' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005  --batch_size=64 --TopK_1=32 --memory_error=True --pos_enc=16  --split_mode=high  --bb=gin  --device=cuda:1

python /main/main_topology_imbalance.py--dataset='NCI1' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005   --batch_size=64 --TopK_2=16 --TopK_1=64  --split_mode=high  --bb=gin  --device=cuda:7

python /main/main_topology_imbalance.py --dataset='REDDIT-BINARY' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005  --batch_size=64 --TopK_2=16 --TopK_1=32  --memory_error=True --split_mode=high  --bb=gin  --device=cuda:7

python /main/main_topology_imbalance.py--dataset='COLLAB' --head=4 --Prototypes=32 --lr=0.001 --weight_decay=0.0005  --batch_size=128 --TopK_2=16 --TopK_1=64  --memory_error=True --Pertu=0 --split_mode=high  --bb=gin --pos_enc=32  --device=cuda:7

python /main/main_topology_imbalance.py--dataset='IMDB-MULTI' --head=4 --Prototypes=16 --lr=0.001 --weight_decay=0.0005  --batch_size=32 --TopK_2=8 --TopK_1=32  --Pertu=0 --memory_error=True --split_mode=high  --pos_enc=16   --bb=gin --Pertu=0  --device=cuda:0



