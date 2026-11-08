import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import random
from san_parser import *

EPOCH = 100

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.accumulator = nn.EmbeddingBag(769, 1, mode='sum', padding_idx=768)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, indices):
        return self.accumulator(indices) + self.bias


def main():
    files_at_once = 4
    batch_size = 10_000
    model = Net()
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    train = ds.dataset('2016_fen_eval/train', format='parquet')
    scan = train.to_batches(
        columns = ['fen', 'eval'],
        filter = (ds.field('ply') >= 12) & (ds.field('eval').is_valid())
    )
    val_fens = set(ds.dataset('2016_fen_eval/val', format='parquet').to_table(columns=['fen'])
                   .column('fen').to_pylist())
    #loop through to get data
    for epoch in range(EPOCH):
        random.shuffle(paths)
        for g in range(0, len(paths), files_at_once):
            group = paths[g:g + files_at_once]
            df = pd.concat([pd.read_parquet(p) for p in group])
            df = df.sample(frac=1)
            for start in range(0, len(df), batch_size):
                batch = df.iloc[start:start + batch_size]
                for x, y in batch:
                    optimizer.zero_grad()
                    pred = model(x)
                    loss = loss_fn(pred, y)
                    loss.backward()
                    optimizer.step()
        
    