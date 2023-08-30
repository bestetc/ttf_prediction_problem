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
        labels = torch.tensor(self.target[df_idx + self.window_size], dtype=torch.float32)
        return data, labels

    def _create_index(self):
        index_count = 0
        for idx in range(self.df.shape[0] - self.window_size):
            # print(self.df['id'][idx:idx + self.window_size].unique().shape[0])
            if self.df['id'][idx:idx + self.window_size].unique().shape[0] == 1:
                self.index[index_count] = idx
                index_count += 1
            # else:
                # print(self.df['id'][idx:idx + self.window_size].unique())


def create_dataloader(df, scaled_data, target, batch_size=16, shuffle=True, drop_last=True,
                      num_workers=0, window_size=30):
    torchdataset = TorchDataset(df=df, scaled_data=scaled_data, target=target, window_size=window_size)
    return DataLoader(torchdataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      drop_last=drop_last), torchdataset

