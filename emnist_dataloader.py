from torch.utils.data import DataLoader
from Load_EMNIST_letters import (
    load_emnist_letters,
    heterogeneous_randomsized_split_data,
    generate_normal_sizes,
)

def emnist_letters_get_dataloader(
    n_parties,
    batch_size,
    alpha=0.5,
    root="./data",
    size_mean=None,
    size_std=None,
    seed=None,
):
    """
        train_local_dls: List[DataLoader]
        test_dl: DataLoader
        net_dataidx_map: Dict[int, List[int]]
    """

    trainds, valds, testds = load_emnist_letters(val_portion=0.4, root=root)

    if size_mean is None:
        size_mean = len(trainds) / n_parties
    if size_std is None:
        size_std = 0.5 * size_mean

    subset_sizes = generate_normal_sizes(
        n=n_parties,
        mean=size_mean,
        std=size_std,
        seed=seed,
    )

    client_subsets = heterogeneous_randomsized_split_data(
        n=n_parties,
        dataset=trainds,
        alpha=alpha,
        subset_sizes=subset_sizes,
        seed=seed,
    )

    # construct train_local_dls: every client has a dataloader
    train_local_dls = []
    net_dataidx_map = {}

    for cid, subset in enumerate(client_subsets):
        if hasattr(subset, "indices"):
            net_dataidx_map[cid] = list(subset.indices)
        else:
            net_dataidx_map[cid] = list(range(len(subset)))

        loader = DataLoader(
            dataset=subset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        train_local_dls.append(loader)

    # construct test dataloader
    test_dl = DataLoader(
        dataset=testds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_local_dls, test_dl, net_dataidx_map
