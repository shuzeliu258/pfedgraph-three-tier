#!/usr/bin/env bash
python pfedgraph_cosine_avg_three-tier_3.py \
  --gpu 2 \
  --dataset cifar_pickle \
  --partition noniid-skew \
  --n_parties 30 \
  --num_local_iterations 200 \
  --comm_round 40 \
  --beta 0.1 \
  --alpha 0.8 \
  --lam 0.1 \
  --num_servers 5 \
  --server_avg_mode weighted \
  --global_server_eval \
  --trainer_speeds "[1.0, 0.5, 2.5, 2.0, 3.0, 2.0, 2.5, 3.0, 0.5, 0.5, 3.0, 3.0, 3.0, 2.5, 2.0, 2.5, 3.5, 3.5, 3.0, 3.5, 2.0, 3.5, 2.0, 1.0, 0.5, 2.0, 3.5, 1.0, 3.5, 3.5]" \
  --edge_speeds "[1.0, 1.0, 1.0, 1.0, 1.0]" \
  --global_speed "0.5" \
  --pickle_name "30_trainer_datast_list_CIFAR_hete_V6.pickle" \