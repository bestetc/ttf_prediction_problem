import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TorchDataset(Dataset):
    def __init__(self, df, scaled_data, target, window_size=30):
        self.df = df
        self.scaled_data = np.array(scaled_data)
        self.target = np.array(target)
        self.window_size = window_size
        self.index = dict()
        self._create_index()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        df_idx = self.index[idx]
        data = torch.tensor(self.scaled_data[df_idx:df_idx + self.window_size, :],  dtype=torch.float32)
        labels = torch.tensor(self.target[df_idx + self.window_size - 1], dtype=torch.float32)
        return data, labels, df_idx

    def _create_index(self):
        index_count = 0
        for idx in range(self.df.shape[0] - self.window_size + 1):
            if self.df['id'][idx:idx + self.window_size].unique().shape[0] == 1:
                self.index[index_count] = idx
                index_count += 1


def create_dataloader(df, scaled_data, target, batch_size=16, shuffle=True, drop_last=True, num_workers=0,
                      window_size=30):
    torchdataset = TorchDataset(df=df, scaled_data=scaled_data, target=target, window_size=window_size)
    return DataLoader(torchdataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last)


def train(model, n_epochs, train_dataloader, optimizer, criterion, test_dataloader=None, verbose=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(1, n_epochs + 1):
        start_time = time.time()
        model.train()
        loss_list = []
        for data, target, _ in train_dataloader:
            data = data.to(device)
            target = target.to(device)
            predicted = model(data)
            loss = criterion(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.detach().cpu().mean())
        train_loss = np.mean(loss_list)

        model.eval()
        if test_dataloader is not None:
            test_loss_list = []
            for data, target, _ in test_dataloader:
                data = data.to(device)
                target = target.to(device)
                predicted = model(data)
                test_loss_list.append(criterion(predicted, target).detach().cpu().mean())
            test_loss = np.mean(np.array(test_loss_list))

        if verbose:
            text = f'epoch: {epoch}/{n_epochs}, train_loss: {train_loss:.3f},'
            if test_dataloader is not None:
                text += f'test_loss: {test_loss:.3f}, '
            text += f'Time: {time.time() - start_time:.3f}'
            print(text)


def predict(model, dataloader, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    len_ds = len(dataloader.dataset.index)
    model.eval()
    predicted, ds_index = np.empty(len_ds), np.empty(len_ds, dtype=int)
    for i, (data, target, idx) in enumerate(dataloader):
        i = np.arange(i * batch_size, i * batch_size + len(idx))
        data = data.to(device)
        model_predict = model(data).detach().cpu().squeeze()
        predicted[i] = model_predict
        ds_index[i] = idx
    return predicted, ds_index
