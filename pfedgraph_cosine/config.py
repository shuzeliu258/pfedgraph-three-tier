import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="1")
    parser.add_argument('--model', type=str, default='simplecnn', help='neural network used in training')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset used for training')
    parser.add_argument('--partition', type=str, default='noniid-skew', help='the data partitioning strategy')
    parser.add_argument('--num_local_iterations', type=int, default=400, help='number of local iterations')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--n_parties', type=int, default=10, help='number of workers in a distributed cluster')
    parser.add_argument('--comm_round', type=int, default=50, help='number of maximum communication roun')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--beta', type=float, default=0.5,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--skew_class', type=int, default = 2, help='The parameter for the noniid-skew for data partitioning')
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='how many clients are sampled in each round')
    parser.add_argument('--concen_loss', type=str, default='uniform_norm', choices=['norm', 'uniform_norm'], help='How to measure the modle difference')
    parser.add_argument('--weight_norm', type=str, default='relu', choices=['sum', 'softmax', 'abs', 'relu', 'sigmoid'], help='How to measure the modle difference')
    parser.add_argument('--difference_measure', type=str, default='all', help='How to measure the model difference')
    
    parser.add_argument('--alpha', type=float, default=0.8, help='Hyper-parameter to avoid concentration')
    parser.add_argument('--lam', type=float, default=0.01, help="Hyper-parameter in the objective")

    # attack
    parser.add_argument('--attack_type', type=str, default="inv_grad")
    parser.add_argument('--attack_ratio', type=float, default=0.0)

    # fedavg
    parser.add_argument('--post_fedavg', type=str, default='none',
                        choices=['none', 'weighted', 'uniform'],
                        help="Do an extra FedAvg after graph aggregation: 'weighted' by data size or 'uniform' (unweighted).")
    parser.add_argument('--post_fedavg_eval', action='store_true',
                        help="After post-FedAvg, evaluate the global model once on the global test set.")

    # three-tier
    parser.add_argument('--num_servers', type=int, default=2,
                        help='Number of upper-layer servers in the hierarchy.')
    parser.add_argument('--global_server_eval', action='store_true',
                        help='Evaluate the global-server averaged model each round (no broadcast).')
    parser.add_argument('--server_avg_mode', type=str, default='uniform',
                        choices=['uniform', 'weighted'],
                        help="How to average clients into a server model: 'uniform' or 'weighted' by data size.")
    parser.add_argument('--client_steps', type=str, default='',
                        help="Steps run by the client, input like \"[100,150,200]\" or \"100,150,200\"")

    # speed
    parser.add_argument(
        '--trainer_speeds',
        type=str,
        default='',
        help=(
            "Speed of each client (trainer). "
            "Format: \"[3.0,1.5,2.5]\" or \"3.0,1.5,2.5\". "
            "Length must match number of clients (n_parties)."
        )
    )

    parser.add_argument(
        '--edge_speeds',
        type=str,
        default='',
        help=(
            "Speed of each edge server. "
            "Format: \"[2.0,1.0,0.5]\" or \"2.0,1.0,0.5\". "
            "Length must match number of edge servers (num_servers)."
        )
    )

    parser.add_argument(
        '--global_speed',
        type=str,
        default='1.0',
        help=(
            "Speed of the global server. "
            "Format: \"1.0\" (single float). "
            "Used to control how frequently global server performs evaluation/aggregation."
        )
    )

    parser.add_argument('--pickle_name', type=str, default='',
                        help="name of pickle")

    args = parser.parse_args()
    cfg = dict()
    cfg["comm_round"] = args.comm_round
    cfg["optimizer"] = args.optimizer
    cfg["lr"] = args.lr
    cfg["epochs"] = args.epochs
    if args.dataset in {'mnist', 'cifar10', 'svhn', 'fmnist', 'yahoo_answers','cifar_pickle'}:
        cfg['classes_size'] = 10
    elif args.dataset == 'cifar100':
        cfg['classes_size'] = 100
    elif args.dataset == 'tinyimagenet':
        cfg['classes_size'] = 200
    elif args.dataset == 'emnist_letters':
        cfg['classes_size'] = 26
    cfg['client_num'] = args.n_parties
    cfg['model_name'] = args.model
    cfg['self_wight'] = 'loss'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args , cfg