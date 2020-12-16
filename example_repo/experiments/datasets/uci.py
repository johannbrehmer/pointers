import numpy as np
import torch
from torch.utils.data import Dataset


def get_uci_dataset(dataset, partition, directory):
    """ Experiments using the UCI datasets, preprocessed as in the MAF and NSF paper, available from https://zenodo.org/record/1161203#.Wmtf_XVl8eN """

    if partition == "train_val":
        return UCIDataset(directory / f"{dataset}_train.npy"), UCIDataset(directory / f"{dataset}_val.npy")
    elif partition == "test":
        return UCIDataset(directory / f"{dataset}_test.npy")
    else:
        raise ValueError(f"Unknown partition {partition}")


class UCIDataset(Dataset):
    def __init__(self, filename):
        self.data = torch.tensor(np.load(filename).astype(np.float32))
        self.n, self.dim = self.data.size()

        super().__init__()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.n
