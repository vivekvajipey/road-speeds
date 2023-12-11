#!/bin/bash

# Activate the Python environment
source road_env/bin/activate

# No Sequence Model: Fully Connected Layer
python nn_train.py --model resnet50 --pretrain imagenet --hidden_dim 128 --batch_size 32 --lr 0.003 --dropout_rate 0.1 --weight_decay 0.0001 --n_epochs 50

# Sequence Model: No Pretraining
echo "Sequence Model: No Pretraining"
python seq_nn_train.py --model resnet50 --pretrain None --hidden_dim 128 --batch_size 32 --lr 0.003 --dropout_rate 0.1 --weight_decay 0.0001 --n_epochs 50

# Sequence Model: Pretraining with ImageNet
echo "Sequence Model: Pretraining with ImageNet"
python seq_nn_train.py --model resnet50 --pretrain imagenet --hidden_dim 128 --batch_size 32 --lr 0.003 --dropout_rate 0.1 --weight_decay 0.0001 --n_epochs 50

# Best Sequence Model: Pretraining with Sentinel-2 self-supervised, 
echo "Sequence Model: Pretraining with Sentinel-2 self-supervised"
python seq_nn_train.py --model resnet50 --pretrain sentinel2 --hidden_dim 128 --batch_size 32 --lr 0.003 --dropout_rate 0.1 --weight_decay 0.0001 --n_epochs 50

# Sequence Model: Pretraining with Sentinel-2 self-supervised using GRU
python seq_nn_train.py --model resnet50 --pretrain imagenet --hidden_dim 128 --rnn_type GRU --batch_size 32 --lr 0.003 --dropout_rate 0.1 --weight_decay 0.0001 --n_epochs 50

echo "All experiments completed."