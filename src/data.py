import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, random_split


class Loader:


    def __init__(self, trn_perc, batch_sz, shuffle=True, val_batching=False):
        self.trn_perc = trn_perc
        self.batch_sz = batch_sz
        self.shuffle = shuffle
        self.val_batch = val_batching

        self.dls = None


    @staticmethod
    def idx_as_list(x):
        if isinstance(x, str):
            raise TypeError("indices must not be a string")
        try:
            return [int(i) for i in x]
        except TypeError:
            return [int(x)]


    def splitter(self, x, y):
        ds = list(zip(x, y))
        if self.shuffle:
            trn_ds, val_ds = random_split(ds, [self.trn_perc, 1 - self.trn_perc])
        else:
            trn_sz = int(self.trn_perc * len(ds))
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