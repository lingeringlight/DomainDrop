#!/bin/bash

device=0
data='PACS'
network='resnet18'

for t in `seq 0 4`
do
  for domain in `seq 0 3`
  do
    python train_domain.py \
    --target $domain \
    --device $device \
    --network $network \
    --time $t \
    --batch_size 64 \
    --data $data \
    --data_root "/data/DataSets/" \
    --result_path "/data/save/models/" \
    --KL_Loss 1 \
    --KL_Loss_weight 1.5 \
    --KL_Loss_T 5 \
    --layer_wise_prob 0.8 \
    --domain_discriminator_flag 1 \
    --domain_loss_flag 1 \
    --discriminator_layers 1 2 3 4 \
    --grl 1 \
    --lambd 0.25 \
    --drop_percent 0.33 \
    --recover_flag 1 \
    --epochs 50 \
    --learning_rate 0.002
  done
done
