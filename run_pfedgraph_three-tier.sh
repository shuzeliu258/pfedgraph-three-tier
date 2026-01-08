#!/usr/bin/env bash
python pfedgraph_cosine_avg_three-tier.py \
  --gpu 2 \
  --dataset cifar10 \
  --partition noniid-skew \
  --n_parties 10 \
  --num_local_iterations 200 \
  --comm_round 100 \
  --beta 0.1 \
  --alpha 0.8 \
  --lam 0.1 \
  --num_servers 2 \
  --server_avg_mode weighted \
  --global_server_eval \
  --client_steps "[100,120,140,160,180,200,220,240,260,280]"
