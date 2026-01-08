import pickle
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

def cifar10_get_dataloader_from_pickle(
    n_parties,
    batch_size,
    alpha=0.5,
    root="./data",
    size_mean=None,
    size_std=None,
    seed=None,
    pickle_path="./cifar_client_subsets.pkl"
):
    """
    return:
        train_local_dls: List[DataLoader]
        test_dl: DataLoader
        net_dataidx_map: Dict[int, List[int]]
    """

    print("Loading CIFAR-10 client subsets from:", pickle_path)

    # read pickle
    with open(pickle_path, "rb") as f:
        client_subsets = pickle.load(f)

    assert len(client_subsets) == n_parties, \
        f"pickle has {len(client_subsets)} clients, but we need {n_parties}"

    # load test dataset
    stats = ((0.4914, 0.4822, 0.4465),
             (0.2023, 0.1994, 0.2010))

    test_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    testds = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=test_tfms
    )

    # construct client dataloader
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

    print("CIFAR10 dataloaders loaded from pickle finished.")

    return train_local_dls, test_dl, net_dataidx_map
