#!/bin/bash

python pfedgraph_cosine.py \
  --gpu 0 \
  --dataset cifar10 \
  --partition noniid-skew \
  --n_parties 10 \
  --num_local_iterations 200 \
  --comm_round 50 \
  --beta 0.1 \
  --alpha 0.8 \
  --lam 0.1
