#!/bin/bash
set -e  # 有报错就退出
mkdir -p logs

# 4 个不同的 pickle_name
pickle_names=(
  "30_trainer_datast_list_CIFAR_hete_V6.pickle"
  "30_trainer_datast_list_CIFAR_hete_V7.pickle"
  "30_trainer_datast_list_CIFAR_hete_V8.pickle"
  "30_trainer_datast_list_CIFAR_hete_V9.pickle"
)

# 原始 trainer_speeds
base_speeds='[1.0, 0.5, 2.5, 2.0, 3.0, 2.0, 2.5, 3.0, 0.5, 0.5, 3.0, 3.0, 3.0, 2.5, 2.0, 2.5, 3.5, 3.5, 3.0, 3.5, 2.0, 3.5, 2.0, 1.0, 0.5, 2.0, 3.5, 1.0, 3.5, 3.5]'

# ×2 之后的 trainer_speeds
double_speeds='[2.0, 1.0, 5.0, 4.0, 6.0, 4.0, 5.0, 6.0, 1.0, 1.0, 6.0, 6.0, 6.0, 5.0, 4.0, 5.0, 7.0, 7.0, 6.0, 7.0, 4.0, 7.0, 4.0, 2.0, 1.0, 4.0, 7.0, 2.0, 7.0, 7.0]'

############################
# 前四遍：原 speeds
############################
for i in {0..3}; do
  pn="${pickle_names[$i]}"
  run_id=$((i+1))
  log_file="logs/run${run_id}_$(basename "${pn%.pickle}")_base.log"

  echo "=============================="
  echo "Run ${run_id}/8  (base speeds)"
  echo "pickle_name = ${pn}"
  echo "log         = ${log_file}"
  echo "=============================="

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
    --trainer_speeds "${base_speeds}" \
    --edge_speeds "[1.0, 1.0, 1.0, 1.0, 1.0]" \
    --global_speed "0.5" \
    --pickle_name "${pn}" \
    > "${log_file}" 2>&1

  echo "Done run ${run_id}"
  echo
done

############################
# 后四遍：×2 speeds
############################
for i in {0..3}; do
  pn="${pickle_names[$i]}"
  run_id=$((i+5))  # 5~8
  log_file="logs/run${run_id}_$(basename "${pn%.pickle}")_x2.log"

  echo "=============================="
  echo "Run ${run_id}/8  (double speeds)"
  echo "pickle_name = ${pn}"
  echo "log         = ${log_file}"
  echo "=============================="

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
    --trainer_speeds "${double_speeds}" \
    --edge_speeds "[1.0, 1.0, 1.0, 1.0, 1.0]" \
    --global_speed "0.5" \
    --pickle_name "${pn}" \
    > "${log_file}" 2>&1

  echo "Done run ${run_id}"
  echo
done
