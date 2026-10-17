import torch
import torch.nn.functional as F


class NNInterpolator:

    def __init__(self, nlay, lay_sz, lr, nepochs):
        self.nlay = nlay
        self.lay_sz = lay_sz
        self.lr = lr
        self.nepochs = nepochs

        self.layers = None
        self.consts = None
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

        self.x_mean = None
        self.x_std = None
        self.y_mean = None
        self.y_std = None


    @staticmethod
    def _stack(dl):
        xs, ys = zip(*[(xb.float(), yb.float()) for xb, yb in dl])
        return torch.cat(xs), torch.cat(ys)


    def coeffs(self):
        return self.layers, self.consts


    def init_coeffs(self, inp_sz, out_sz):
        sizes = [inp_sz] + (self.nlay - 1) * [self.lay_sz] + [out_sz]
        n = len(sizes)
        self.layers = [
            torch.randn((sizes[i], sizes[i+1])) * (2 / sizes[i]) ** 0.5
            for i in range(n-1)
        ]
        self.consts = [
            (torch.rand(sizes[i+1]) * 2 - 1) * (1 / sizes[i]) ** 0.5
            for i in range(n-1)
        ]
        for l in self.layers + self.consts:
            l.requires_grad_()


    def calc_preds(self, x):
        res = x
        n = len(self.layers)
        for i,l in enumerate(self.layers):
            res = res @ l + self.consts[i]
            if i != n-1:
                res = F.relu(res)
        return res


    @staticmethod
    def calc_loss(preds, y):
        return ((preds - y)**2).mean(0)


    def update_coeffs(self):
        for w in self.layers + self.consts:
            w.sub_(w.grad * self.lr)
            w.grad.zero_()


    def epoch(self, dls):
        trn_dl, val_dl = dls

        trn_total, trn_n = 0, 0
        for xb, yb in trn_dl:
            xb = (xb - self.x_mean) / self.x_std
            yb = (yb - self.y_mean) / self.y_std

            preds = self.calc_preds(xb)
            loss = self.calc_loss(preds, yb)
            loss.backward()
            with torch.no_grad():
                self.update_coeffs()

            trn_total += loss.item() * len(xb)
            trn_n += len(xb)
        train_loss = trn_total / trn_n

        val_total, val_n = 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb = (xb - self.x_mean) / self.x_std
                yb = (yb - self.y_mean) / self.y_std

                preds = self.calc_preds(xb)
                loss = self.calc_loss(preds, yb)

                val_total += loss.item() * len(xb)
                val_n += len(xb)
        val_loss = val_total / val_n

        return train_loss, val_loss


    def fit(self, dls):
        trn_dl, _ = dls

        x_trn, y_trn = self._stack(trn_dl)
        self.x_mean, self.x_std = x_trn.mean(0), x_trn.std(0).clamp_min(1e-12)
        self.y_mean, self.y_std = y_trn.mean(0), y_trn.std(0).clamp_min(1e-12)

        xb, yb = next(iter(trn_dl))
        self.init_coeffs(xb.shape[1], yb.shape[1])

        for i in range(self.nepochs):
            train_loss, val_loss = self.epoch(dls)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

        if (torch.isnan(torch.tensor(self.history['train_loss'])).any() or
            torch.isnan(torch.tensor(self.history['val_loss'])).any() or
            torch.isinf(torch.tensor(self.history['train_loss'])).any() or
            torch.isinf(torch.tensor(self.history['val_loss'])).any()
        ):
            raise ValueError("The training process has diverged. Try a different lr.")

        return self.history


    def predict(self, x):
        if self.layers is None or self.x_mean is None:
            raise RuntimeError(
                "NNInterpolator has not been trained yet. Call fit(data) first."
                )
        
        x = (x - self.x_mean) / self.x_std
        with torch.no_grad():
            preds = self.calc_preds(x)
            return preds * self.y_std + self.y_mean