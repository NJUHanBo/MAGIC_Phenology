import numpy as np
import torch
import torch.utils.data as data
import os
from datasets import PARENT_DIR


class TimeSeriesSTSC(data.Dataset):
    """
    多波段 STSC 数据集。
    npy shape: (N, 7, 46) → 返回展平向量 (322,) 以兼容现有 MLP FeatureExtractor。
    """

    def __init__(self, npy_path):
        super().__init__()
        full = os.path.join(PARENT_DIR, npy_path) if not os.path.isabs(npy_path) else npy_path
        arr = np.load(full)
        if arr.ndim != 3 or arr.shape[1:] != (7, 46):
            raise ValueError(f"Expected (N,7,46), got {arr.shape} from {full}")
        self.x = arr.reshape(arr.shape[0], -1).astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return {"stsc": torch.from_numpy(self.x[index])}
