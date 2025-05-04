import torch
from torch.utils.data import Dataset

class LOBDataset(Dataset):
    def __init__(self, X, y, mode=False):
        self.X = torch.tensor(X, dtype=torch.float32) # (B, 100, 40) for transLOB
        if mode:
            self.X= self.X.unsqueeze(1) # (B, 1, 100, 40) for deepLOB
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]