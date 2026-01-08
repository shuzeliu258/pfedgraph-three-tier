import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # sliu11

import copy
import math
import random
import time
from test import compute_acc
import ast

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from pfedgraph_cosine.config import get_args
from pfedgraph_cosine.utils import aggregation_by_graph, update_graph_matrix_neighbor
from model import simplecnn, textcnn, emnist_letters_cnn
from prepare_data import get_dataloader
from attack import *
# from emnist_dataloader import emnist_letters_get_dataloader
from cifar_dataloader_from_pickle import cifar10_get_dataloader_from_pickle
# 统计client准确率；难度：9>8>6>7；5个edge server[2,4,6,8,10]；client以一倍速和二倍速分别测试；client从自己的train data选出30%测试
N_trainers = [2, 4, 6, 8, 10]   # 每个 server 拥有的 client 数目


def local_train_pfedgraph(
    args,
    round,
    nets_this_round,
    cluster_models,
    train_local_dls,
    local_test_dl,
    benign_client_list,
    client_steps=None,
):
    for net_id, net in nets_this_round.items():

        train_local_dl = train_local_dls[net_id]
        local_test_dl = local_test_dls[net_id]

        # ---- Optimizer & Loss ----
        if args.optimizer == "adam":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=args.lr,
                weight_decay=args.reg,
            )
        elif args.optimizer == "amsgrad":
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=args.lr,
                weight_decay=args.reg,
                amsgrad=True,
            )
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, net.parameters()),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.reg,
            )
        criterion = torch.nn.CrossEntropyLoss()

        if round > 0:
            cluster_model = cluster_models[net_id].cuda()

        net.cuda()
        net.train()
        iterator = iter(train_local_dl)

        local_steps = (
            client_steps[net_id]
            if client_steps is not None
            else args.num_local_iterations
        )

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
                loss2 = (
                    args.lam
                    * torch.dot(cluster_model, flatten_model)
                    / torch.linalg.norm(flatten_model)
                )
                loss2.backward()

            loss.backward()
            optimizer.step()

        net.eval()
        local_acc = compute_acc(net, local_test_dl)
        print(
            f">> Client {net_id} | Round {round} | "
            f"Local Test Acc (30% of its train-data): {local_acc:.5f}"
        )

    return


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

def mask_graph_matrix_by_edge(graph_matrix, client2edge):
    """
    preserve edges between clients within the same edge server, set edges crossing edge servers to 0, and perform row normalization.
    """
    with torch.no_grad():
        num_clients = graph_matrix.shape[0]

        for i in range(num_clients):
            for j in range(num_clients):
                if client2edge[i] != client2edge[j]:
                    graph_matrix[i, j] = 0.0

        graph_matrix.fill_diagonal_(0.0)

        row_sums = graph_matrix.sum(dim=1, keepdim=True)  # shape (N,1)
        nonzero_rows = (row_sums > 0).squeeze(1)

        graph_matrix[nonzero_rows] = graph_matrix[nonzero_rows] / row_sums[nonzero_rows]

        zero_rows = ~nonzero_rows
        if zero_rows.any():
            graph_matrix[zero_rows] = 0.0
            for idx in torch.nonzero(zero_rows, as_tuple=False).view(-1):
                graph_matrix[idx, idx] = 1.0

    return graph_matrix


# ==================== main ====================
all_global_acc = []

args, cfg = get_args()
print(args)

seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

pickle_name = args.pickle_name

# ==== load trainer_speeds / edge_speeds / global_speed, and generate client_steps ====

trainer_speeds = None
edge_speeds = None
global_speed = None
client_steps = None
edge_freq = None
global_freq = 1

# ---- 1) load trainer_speeds, and generate client_steps ----
if getattr(args, "trainer_speeds", None):
    try:
        trainer_speeds = ast.literal_eval(args.trainer_speeds)
    except Exception as e:
        raise ValueError(f"Unable to parse --trainer_speeds: {e}")

    if not isinstance(trainer_speeds, list):
        raise ValueError("--trainer_speeds must be list，such as \"[3.0,1.5,...]\"")

    assert len(trainer_speeds) == args.n_parties, (
        f"trainer_speeds length is  {len(trainer_speeds)}, but args.n_parties={args.n_parties}"
    )

    print(f"[INFO] trainer_speeds: {trainer_speeds}")

    max_speed = max(trainer_speeds)
    # base_client_step, default 200
    base_step = getattr(args, "base_client_step", 200)

    client_steps = [
        max(1, int(base_step * (s / max_speed))) for s in trainer_speeds
    ]
    print(f"[INFO] Auto-generated client_steps from trainer_speeds: {client_steps}")

# ---- 2) no trainer_speeds, using original client_steps ----
else:
    if args.client_steps:
        try:
            if args.client_steps.strip().startswith("["):
                client_steps = ast.literal_eval(args.client_steps)
            else:
                client_steps = [int(x) for x in args.client_steps.split(",")]
        except Exception as e:
            raise ValueError(f"Unable to resolve --client_steps parameter: {e}")

        if len(client_steps) != args.n_parties:
            raise ValueError(
                f"--client_steps 长度为 {len(client_steps)}, 但 args.n_parties={args.n_parties}"
            )
        print(f"[INFO] clients steps (from args.client_steps): {client_steps}")
    else:
        # all clients have same steps.
        client_steps = [args.num_local_iterations] * args.n_parties
        print(f"[INFO] clients steps (uniform): {client_steps}")

# ---- 3) load edge_speeds, generate edge_freq ----
if getattr(args, "edge_speeds", None):
    try:
        edge_speeds = ast.literal_eval(args.edge_speeds)
    except Exception as e:
        raise ValueError(f"Unable to parse --edge_speeds: {e}")

    if not isinstance(edge_speeds, list):
        raise ValueError("--edge_speeds must be list, such as \"[2.0,1.0,...]\"")

    assert len(edge_speeds) == args.num_servers, (
        f"edge_speeds length is {len(edge_speeds)}, but args.num_servers={args.num_servers}"
    )
    print(f"[INFO] edge_speeds: {edge_speeds}")

    max_edge_speed = max(edge_speeds)
    # frequency = max_speed / current speed
    edge_freq = [
        max(1, int(round(max_edge_speed / s))) for s in edge_speeds
    ]
    print(f"[INFO] edge aggregation frequency per server: {edge_freq}")
else:
    edge_freq = [1 for _ in range(args.num_servers)]
    print(f"[INFO] edge_freq (default=1 for all): {edge_freq}")

# ---- 4) load global_speed ----
if getattr(args, "global_speed", None) is not None:
    try:
        global_speed = float(args.global_speed)
    except Exception as e:
        raise ValueError(f"Unable to parse --global_speed: {e}")

    if global_speed <= 0:
        raise ValueError("--global_speed 必须 > 0")

    # freq means "how many rounds to perform a global evaluation".
    global_freq = max(1, int(round(1.0 / global_speed)))
    print(f"[INFO] global_speed={global_speed}, global_freq={global_freq}")
else:
    global_freq = 1
    print(f"[INFO] global_speed not set, use global_freq=1")



n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(
    party_list, int(args.n_parties * (1 - args.attack_ratio))
)
benign_client_list.sort()
print(f">> -------- Benign clients: {benign_client_list} --------")

# load CIFAR10 from pickle
train_local_dls, test_dl, net_dataidx_map = cifar10_get_dataloader_from_pickle(
    n_parties=args.n_parties,
    batch_size=args.batch_size,
    alpha=args.beta,
    root=args.datadir,
    seed=args.init_seed,
    pickle_path=pickle_name,
)

# every client 70% train, 30% test
def split_client_train_test(loaders, test_ratio=0.3):
    new_train_loaders = []
    local_test_loaders = []
    for dl in loaders:
        dataset = dl.dataset
        n_total = len(dataset)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_test

        train_ds, test_ds = random_split(dataset, [n_train, n_test])

        bs = dl.batch_size
        new_train_loaders.append(
            DataLoader(train_ds, batch_size=bs, shuffle=True)
        )
        local_test_loaders.append(
            DataLoader(test_ds, batch_size=bs, shuffle=False)
        )
    return new_train_loaders, local_test_loaders

train_local_dls, local_test_dls = split_client_train_test(train_local_dls, test_ratio=0.3)

if args.dataset == "cifar10":
    model = simplecnn
elif args.dataset == "cifar100":
    model = simplecnn
elif args.dataset == "cifar_pickle":
    model = simplecnn
elif args.dataset == "yahoo_answers":
    model = textcnn
elif args.dataset == "emnist_letters":
    model = emnist_letters_cnn

global_model = model(cfg["classes_size"])
global_parameters = global_model.state_dict()
local_models = []
dw = []
for i in range(cfg["client_num"]):
    local_models.append(model(cfg["classes_size"]))
    dw.append(
        {key: torch.zeros_like(value) for key, value in local_models[i].named_parameters()}
    )

graph_matrix = torch.ones(len(local_models), len(local_models)) / (
    len(local_models) - 1
)  # Collaboration Graph
graph_matrix[range(len(local_models)), range(len(local_models))] = 0

for net in local_models:
    net.load_state_dict(global_parameters)


def make_server_groups(num_servers, client_ids):
    groups = []
    per = math.ceil(len(client_ids) / num_servers)
    for i in range(num_servers):
        block = client_ids[i * per : (i + 1) * per]
        if block:
            groups.append(block)
    return groups

def make_server_groups_by_ntrainers(N_trainers, n_parties):
    if sum(N_trainers) != n_parties:
        raise ValueError(
            f"N_trainers\' sum {sum(N_trainers)} must be equal to n_parties={n_parties}"
        )

    server_groups = []
    cid = 0
    for num in N_trainers:
        server_groups.append(list(range(cid, cid + num)))
        cid += num

    return server_groups


all_client_ids = list(range(args.n_parties))  # 或 cfg['client_num']
#server_groups = make_server_groups(args.num_servers, all_client_ids)
server_groups = make_server_groups_by_ntrainers(N_trainers, args.n_parties)
print(f">> Server groups: {server_groups}")

# ==== record client's edge server ====
client2edge = {}
for e_idx, clients in enumerate(server_groups):
    for cid in clients:
        client2edge[cid] = e_idx

cluster_model_vectors = {}

for round in range(cfg["comm_round"]):
    print(f"\n========== Round {round} ==========")
    server_states_this_round = []

    # ===== traversal each server (edge) =====
    for s_idx, client_ids in enumerate(server_groups):
        party_list_this_round = client_ids

        nets_this_round = {k: local_models[k] for k in party_list_this_round}
        nets_param_start = {
            k: copy.deepcopy(local_models[k]) for k in party_list_this_round
        }

        # --- local train ---
        local_train_pfedgraph(
            args,
            round,
            nets_this_round,
            cluster_model_vectors,
            train_local_dls,
            local_test_dls,
            benign_client_list,
            client_steps=client_steps,
        )

        total_data_points = sum([len(net_dataidx_map[k]) for k in party_list_this_round])
        fed_avg_freqs = {
            k: len(net_dataidx_map[k]) / total_data_points for k in party_list_this_round
        }
        manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)

        # --- update graph ---
        graph_matrix = update_graph_matrix_neighbor(
            graph_matrix,
            nets_this_round,
            global_parameters,
            dw,
            fed_avg_freqs,
            args.alpha,
            args.difference_measure,
        )

        graph_matrix = mask_graph_matrix_by_edge(graph_matrix, client2edge)

        cluster_vec = aggregation_by_graph(
            cfg, graph_matrix, nets_this_round, global_parameters
        )
        cluster_model_vectors.update(cluster_vec)

        # --- aggregate server's model (edge-level) ---
        if args.server_avg_mode == "weighted":
            server_state = compute_fedavg_params(
                nets_this_round, weight_dict=fed_avg_freqs
            )
        else:
            server_state = compute_fedavg_params(nets_this_round, weight_dict=None)

        # ==== decide uploading according to edge_freq ====
        if edge_freq[s_idx] > 1 and (round % edge_freq[s_idx] != 0):
            print(
                f">> [Edge-Speed] Edge-Server {s_idx} only trains locally in Round {round} and is not uploaded to Global."
                f"(freq={edge_freq[s_idx]})"
            )
        else:
            server_states_this_round.append(server_state)

            # evaluate uploaded edges
            server_eval_model = model(cfg["classes_size"])
            server_eval_model.load_state_dict(server_state)
            server_acc = compute_acc(server_eval_model, test_dl)
            print(
                f">> Edge-Server {s_idx} | Round {round} | "
                f"Server Test Acc (uniform over clients): {server_acc:.5f}"
            )

    # no edge server, skip global aggregation
    if len(server_states_this_round) == 0:
        print(f">> [Edge-Speed] No edge server participated at Round {round}, "
              f"skip global aggregation.")
        print("-" * 80)
        continue

    # ===== Global Server =====
    global_server_state = average_state_dicts(server_states_this_round)

    # ==== global_freq control evaluate and print frequency ====
    if (round % global_freq) == 0 and args.global_server_eval:
        global_eval_model = model(cfg["classes_size"])
        global_eval_model.load_state_dict(global_server_state)
        global_acc = compute_acc(global_eval_model, test_dl)
        print(
            f">> Global-Server | Round {round} | "
            f"Global Test Acc (uniform over servers): {global_acc:.5f}"
        )
        all_global_acc.append(global_acc)
    else:
        if args.global_server_eval:
            print(
                f">> [Global-Speed] Skip Global-Server eval at Round {round} "
                f"(global_freq={global_freq})"
            )
            all_global_acc.append(None)

    print("-" * 80)

# save result
timestamp = int(time.time())
save_path = f"results_global_acc_run_{timestamp}.npy"

np.save(save_path, np.array(all_global_acc))
print(f"[INFO] Saved global accuracy curve to {save_path}")
