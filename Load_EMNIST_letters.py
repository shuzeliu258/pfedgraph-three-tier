import math
from collections import Counter
from torchvision import datasets, transforms
import torch
import torchvision
import random
import matplotlib.pyplot as plt
import pickle
import numpy as np
from torch.utils.data import Subset

class EMNISTLettersWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset
        for attr in dir(base_dataset):
            if not attr.startswith("__"):
                try:
                    setattr(self, attr, getattr(base_dataset, attr))
                except:
                    pass

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return img, label - 1



def load_emnist_letters(val_portion=0.4, root='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # load train data
    trainds = torchvision.datasets.EMNIST(
        root=root,
        split='letters',
        train=True,
        download=True,
        transform=transform
    )

    # load test data
    testds = torchvision.datasets.EMNIST(
        root=root,
        split='letters',
        train=False,
        download=True,
        transform=transform
    )

    trainds = EMNISTLettersWrapper(trainds)
    testds = EMNISTLettersWrapper(testds)

    # split valds
    v = int(val_portion * len(testds))
    valds, _ = torch.utils.data.random_split(testds, [v, len(testds) - v])

    return trainds, valds, testds

def heterogeneous_randomsized_split_data(n,
    dataset,
    alpha=0.5,
    subset_sizes=None,
    seed=None
):
    """
    Split `dataset` into n subsets, sampling with replacement so indices
    can repeat across subsets.  Class‐distribution skew per subset is
    controlled by `alpha` (Dirichlet).  You can also force each subset i
    to have exactly subset_sizes[i] samples.

    Args:
        dataset: must support dataset[i] -> (x, label) and have .class_to_idx.
        n (int): number of subsets.
        alpha (float): Dirichlet concentration for class splits (smaller = more skew).
        subset_sizes (list[int] or None): if provided, len must be n, and each
            subset i will end up with exactly subset_sizes[i] samples.
        seed (int, optional): random seed.

    Returns:
        List[torch.utils.data.Subset]: length-n list of Subsets.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    N = len(dataset)
    n_c = len(dataset.class_to_idx)
    classes = list(range(n_c))

    # 1) group indices by class
    data_by_class = {c: [] for c in classes}
    for idx in range(N):
        _, label = dataset[idx]
        data_by_class[label].append(idx)

    # 2) initial Dirichlet split per class -> integer counts int_counts[c,i]
    int_counts = np.zeros((n_c, n), dtype=int)
    for c in classes:
        idx_list = data_by_class[c]
        M_c = len(idx_list)
        if M_c == 0:
            continue

        random.shuffle(idx_list)
        props = np.random.dirichlet([alpha] * n)
        raw = props * M_c
        base = np.floor(raw).astype(int)
        rem = M_c - base.sum()
        frac = raw - base
        # give the leftover 'rem' ones to the largest fractional parts
        for i in np.argsort(-frac)[:rem]:
            base[i] += 1
        int_counts[c, :] = base

    # 3) if the user supplied subset_sizes, rescale each subset’s class counts
    if subset_sizes is not None:
        if len(subset_sizes) != n:
            raise ValueError("subset_sizes must be length n")
        for i in range(n):
            orig = int_counts[:, i].copy()
            total_orig = orig.sum()
            if total_orig == 0:
                continue  # no classes here
            # preserve class proportions orig/total_orig
            props = orig / total_orig
            target = np.array(subset_sizes[i] * props)
            base = np.floor(target).astype(int)
            rem = subset_sizes[i] - base.sum()
            frac = target - base
            for c in np.argsort(-frac)[:rem]:
                base[c] += 1
            int_counts[:, i] = base

    # 4) sample with replacement per (class c, subset i)
    subsets_indices = [[] for _ in range(n)]
    for c in classes:
        pool = data_by_class[c]
        if not pool:
            continue
        for i in range(n):
            k = int(int_counts[c, i])
            if k > 0:
                # choices with replacement
                picks = random.choices(pool, k=k)
                subsets_indices[i].extend(picks)

    # 5) wrap in Subset objects
    return [Subset(dataset, idxs) for idxs in subsets_indices]

def generate_normal_sizes(n, mean=1000, std=500, seed=None):
    """
    Return a list of n integer sizes sampled from N(mean, std^2).
    Negative draws are clipped to zero.

    Args:
        n (int): how many sizes to generate.
        mean (float): the normal distribution mean.
        std (float): the normal distribution standard deviation.
        seed (int, optional): random seed for reproducibility.

    Returns:
        List[int]: n non-negative integer sizes.
    """
    if seed is not None:
        np.random.seed(seed)
    draws = np.random.normal(loc=mean, scale=std, size=n)
    draws_clipped = np.clip(draws, a_min=0, a_max=None)
    return np.round(draws_clipped).astype(int).tolist()