import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # sliu11

import copy
import math
import random
import time
from test import compute_acc, compute_local_test_accuracy
import ast

import numpy as np
import torch
import torch.optim as optim

from pfedgraph_cosine.config import get_args
from pfedgraph_cosine.utils import aggregation_by_graph, update_graph_matrix_neighbor
from model import simplecnn, textcnn
from prepare_data import get_dataloader
from attack import *

def local_train_pfedgraph(args, round, nets_this_round, cluster_models, train_local_dls, val_local_dls, test_dl, data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list, client_steps=None):
    
    for net_id, net in nets_this_round.items():
        
        train_local_dl = train_local_dls[net_id]
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        if round > 0:
            cluster_model = cluster_models[net_id].cuda()
        
        net.cuda()
        net.train()
        iterator = iter(train_local_dl)
        local_steps = client_steps[net_id] if client_steps is not None else args.num_local_iterations
        for iteration in range(local_steps):
            try:
                x, target = next(iterator)
            except StopIteration:
                iterator = iter(train_local_dl)
                x, target = next(iterator)
            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()
            target = target.long()

            out = net(x)
            loss = criterion(out, target)
        

            if round > 0:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                loss2.backward()
                
            loss.backward()
            optimizer.step()
        
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val_local_dls[net_id])
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, test_dl, data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test2 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc))
        net.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()

def compute_fedavg_params(nets_this_round, weight_dict=None):
    """
    average clients' parameter
    weight_dict=None: average without weight
    weight_dict: dict[client_id -> weight] average with weight
    return: state_dict
    """
    first_key = next(iter(nets_this_round))
    ref_state = nets_this_round[first_key].state_dict()
    avg_state = {k: torch.zeros_like(v, device=v.device) for k, v in ref_state.items()}
    if weight_dict is None:
        scale = 1.0 / len(nets_this_round)
        with torch.no_grad():
            for _, net in nets_this_round.items():
                for name, tensor in net.state_dict().items():
                    avg_state[name] += tensor.detach() * scale
    else:
        s = sum(weight_dict.values())
        assert s > 0, "weight_dict sum must be > 0"
        with torch.no_grad():
            for k, net in nets_this_round.items():
                w = weight_dict[k] / s
                for name, tensor in net.state_dict().items():
                    avg_state[name] += tensor.detach() * w
    for name in avg_state:
        avg_state[name] = avg_state[name].to('cpu')
    return avg_state

def average_state_dicts(state_list):
    """
    average state_dicts
    """
    assert len(state_list) > 0
    out = {}
    for name in state_list[0]:
        acc = None
        for st in state_list:
            t = st[name]
            acc = t if acc is None else acc + t
        out[name] = acc / float(len(state_list))
    return out

# main
args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

if args.client_steps:
    try:
        if args.client_steps.strip().startswith('['):
            client_steps = ast.literal_eval(args.client_steps)
        else:
            client_steps = [int(x) for x in args.client_steps.split(',')]
    except Exception as e:
        raise ValueError(f"Unable to resolve --client_steps parameter: {e}")

    print(f"[INFO] clients steps: {client_steps}")
else:
    # all clients have same steps.
    client_steps = [args.num_local_iterations] * args.num_clients

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
benign_client_list.sort()
print(f'>> -------- Benign clients: {benign_client_list} --------')

train_local_dls, val_local_dls, test_dl, net_dataidx_map, traindata_cls_counts, data_distributions = get_dataloader(args)

if args.dataset == 'cifar10':
    model = simplecnn
elif args.dataset == 'cifar100':
    model = simplecnn
elif args.dataset == 'yahoo_answers':
    model = textcnn
    
global_model = model(cfg['classes_size'])
global_parameters = global_model.state_dict()
local_models = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []
for i in range(cfg['client_num']):
    local_models.append(model(cfg['classes_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 # Collaboration Graph
graph_matrix[range(len(local_models)), range(len(local_models))] = 0

for net in local_models:
    net.load_state_dict(global_parameters)

def make_server_groups(num_servers, client_ids):
    groups = []
    per = math.ceil(len(client_ids) / num_servers)
    for i in range(num_servers):
        block = client_ids[i*per:(i+1)*per]
        if block:
            groups.append(block)
    return groups

all_client_ids = list(range(args.n_parties))  # 或 cfg['client_num']
server_groups = make_server_groups(args.num_servers, all_client_ids)
print(f'>> Server groups: {server_groups}')
'''
server_graphs = []
server_cluster_models = []
for grp in server_groups:
    n = len(grp)
    G = torch.ones(n, n) / (n - 1) if n > 1 else torch.zeros(1, 1)
    if n > 1:
        G[range(n), range(n)] = 0
    server_graphs.append(G)
    server_cluster_models.append({})
'''
cluster_model_vectors = {}

for round in range(cfg["comm_round"]):
    server_states_this_round = []  # 存放每个 server 的代表模型（state_dict）

    # ===== traversal each server =====
    for s_idx, client_ids in enumerate(server_groups):
        party_list_this_round = client_ids

        nets_this_round = {k: local_models[k] for k in party_list_this_round}
        nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}

        mean_personalized_acc = local_train_pfedgraph(
            args, round, nets_this_round, cluster_model_vectors,
            train_local_dls, val_local_dls, test_dl,
            data_distributions, best_val_acc_list, best_test_acc_list, benign_client_list,
            client_steps = client_steps
        )

        total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
        fed_avg_freqs = {k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round}
        manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)

        # --- update graph ---
        graph_matrix = update_graph_matrix_neighbor(
            graph_matrix, nets_this_round, global_parameters, dw, fed_avg_freqs, args.alpha, args.difference_measure
        )
        # cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters)
        cluster_vec = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters)
        cluster_model_vectors.update(cluster_vec)  # 关键：累加，不要覆盖
        # --- aggregate server's model ---
        if args.server_avg_mode == 'weighted':
            server_state = compute_fedavg_params(nets_this_round, weight_dict=fed_avg_freqs)
        else:
            server_state = compute_fedavg_params(nets_this_round, weight_dict=None)
        server_states_this_round.append(server_state)

        server_eval_model = model(cfg['classes_size'])
        server_eval_model.load_state_dict(server_state)
        server_acc = compute_acc(server_eval_model, test_dl)
        print(f'>> Server {s_idx} | Round {round} | Server Test Acc (uniform over clients): {server_acc:.5f}')

    # ===== Global Server =====
    global_server_state = average_state_dicts(server_states_this_round)

    if args.global_server_eval:
        global_eval_model = model(cfg['classes_size'])
        global_eval_model.load_state_dict(global_server_state)
        global_acc = compute_acc(global_eval_model, test_dl)
        print(f'>> Global-Server | Round {round} | Global Test Acc (uniform over servers): {global_acc:.5f}')

    print('-' * 80)

 