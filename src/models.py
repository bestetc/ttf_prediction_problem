import torch
from torch import nn


class TimeSeriesTransformer(nn.Module):
    def __init__(self, d_model=16, window_size=30, nhead=1, num_encoder_layers=2, dim_feedforward=32, dropout=0.2,
                 add_sigmoid=False):
        super().__init__()
        layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                            dropout=dropout,
                                            batch_first=True)
        self.encoder = nn.TransformerEncoder(layers, num_layers=num_encoder_layers)
        self.linear = nn.Linear(d_model * window_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.d_model = d_model
        self.window_size = window_size
        self.add_sigmoid = add_sigmoid

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.d_model * self.window_size)
        x = self.linear(x)
        if self.add_sigmoid:
            x = self.sigmoid(x)
        return x
