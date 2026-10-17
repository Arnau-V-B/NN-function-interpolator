import pandas as pd
import torch
from torch.utils.data import TensorDataset, random_split, Subset, DataLoader


class Loader:


    def __init__(self, trn_perc, batch_sz, shuffle=True, val_batching=False):
        self.trn_perc = trn_perc
        self.batch_sz = batch_sz
        self.shuffle = shuffle
        self.val_batch = val_batching

        self.dls = None

        if not 0 < trn_perc < 1:
            raise ValueError("trn_perc must be between 0 and 1")


    @staticmethod
    def idx_as_list(x):
        if isinstance(x, str):
            raise TypeError("indices must not be a string")
        try:
            return [int(i) for i in x]
        except TypeError:
            return [int(x)]


    def splitter(self, x, y):
        ds = TensorDataset(x, y)

        trn_sz = int(self.trn_perc * len(ds))
        val_sz = len(ds) - trn_sz
        if trn_sz < 1 or val_sz < 1:
            raise ValueError("there must be at leat one observation in both"
                            "training and validation sets after splitting")

        if self.shuffle:
            trn_ds, val_ds = random_split(ds, [self.trn_perc, 1 - self.trn_perc])
        else:
            trn_ds = Subset(ds, range(trn_sz))
            val_ds = Subset(ds, range(trn_sz, len(ds)))
        return trn_ds, val_ds


    def build_dls(self, x, y):
        trn_ds, val_ds = self.splitter(x, y)

        trn_dl = DataLoader(trn_ds, batch_size=self.batch_sz, shuffle=self.shuffle)
        if self.val_batching:
            val_dl = DataLoader(val_ds, batch_size=self.batch_sz, shuffle=self.shuffle)
        else:
            val_dl = DataLoader(val_ds, batch_size=len(val_ds), shuffle=self.shuffle)

        return (trn_dl, val_dl)


    def from_csv(self, path, x_cols, y_cols):
        data = pd.read_csv(path)
        x_cols = self.idx_as_list(x_cols)
        y_cols = self.idx_as_list(y_cols)

        x = torch.tensor(data.iloc[:, x_cols].values)
        y = torch.tensor(data.iloc[:, y_cols].values)

        self.dls = self.build_dls(x, y)

        return self.dls