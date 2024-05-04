from __future__ import annotations
""" データセットを作成するコード """

"""
インポート
"""
import torch
from torch.utils.data import Dataset

class BoardDataset(Dataset):
    """
    盤面情報と勝者、行動のデータセット
    """
    def __init__(self, data: 'list[list[list[int]]]', labels: 'list[tuple[int, int]]'):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = \
        {
            'board': torch.tensor([self.data[idx]], dtype=torch.float),
            'winner': torch.tensor(self.labels[idx][0], dtype=torch.float),
            'action': torch.tensor(self.labels[idx][1], dtype=torch.float),
        }
        return sample