import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import os
from datasets import PARENT_DIR

class TimeSeriesEVI(data.Dataset):
    def __init__(self, csv_path):
        super(TimeSeriesEVI, self).__init__()
        self.data_df = pd.read_csv(os.path.join(PARENT_DIR, csv_path))
        self.evi_cols = [c for c in self.data_df.columns if c.startswith('EVI')]
        self.n_bands = len(self.evi_cols) if self.evi_cols else self.data_df.shape[1]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        sample = self.data_df.iloc[index]
        if self.evi_cols:
            evi_values = sample[self.evi_cols].values.astype('float32')
        else:
            evi_values = sample[:self.n_bands].values.astype('float32')
        return {'evi': torch.tensor(evi_values, dtype=torch.float32)}