#!/bin/sh
################### Train #################

# vaihingen
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29506 train.py --dataset vaihingen --end_epoch 100 --lr 0.0003 --train_batchsize 4 --models swinT --head mlphead --crop_size 512 512 --use_mixup 0 --use_edge 0 --information num1

# potsdam
#CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29507 train.py --dataset potsdam --end_epoch 50 --lr 0.0001 --train_batchsize 4 --models swinT --head mlphead --crop_size 512 512 --use_mixup 0 --use_edge 0 --information num2


################### Test #################

# vaihingen
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29506 test.py --dataset vaihingen --val_batchsize 16 --models swinT --head mlphead --crop_size 512 512 --save_dir work_dir --base_dir ../../ --information num1

# potsdam
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port 29507 test.py --dataset potsdam --val_batchsize 16 --models swinT --head mlphead --crop_size 512 512 --save_dir work_dir --base_dir ../../ --information num2








