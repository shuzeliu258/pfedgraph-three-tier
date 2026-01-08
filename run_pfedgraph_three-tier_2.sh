#!/usr/bin/env bash
python pfedgraph_cosine_avg_three-tier_2.py \
  --gpu 2 \
  --dataset emnist_letters \
  --partition noniid-skew \
  --n_parties 20 \
  --num_local_iterations 200 \
  --comm_round 100 \
  --beta 0.1 \
  --alpha 0.8 \
  --lam 0.1 \
  --num_servers 4 \
  --server_avg_mode weighted \
  --global_server_eval \
  --client_steps "[100,120,140,160,180,200,220,240,260,280,300,220,140,160,280,200,120,240,260,180]"
