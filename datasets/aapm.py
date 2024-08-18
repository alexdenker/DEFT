import os

import numpy as np
import torch
from torch.utils.data import Dataset


class AAPMDataset(Dataset):
    def __init__(self, part: str, base_path: str) -> None:
        assert part in ["val", "test"], "Part must be either 'val' or 'test'."
        self.part = part
        self.base_path = base_path
        self.data = self.load_data(self.part)

    def load_data(self, file_name: str) -> torch.Tensor:
        return torch.from_numpy(
            np.load(os.path.join(self.base_path, file_name + "_data.npy"))
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = self.data[idx][None, ...]
        return x / x.max()


class AAPMDatasetHJ(Dataset):
    def __init__(
        self,
        base_path: str = "/localdata/AlexanderDenker/score_based_baseline/AAPM/256_sorted/256_sorted/L067",
        seed: int = 1,
    ) -> None:
        self.base_path = base_path
        file_list = os.listdir(self.base_path)
        file_list.sort(key=lambda n: float(n.split(".")[0]))
        self.slices = file_list  # file_list[::8]

        # if self.part == 'val':
        # self.slices = list(set(file_list) - set(file_list[::8]))
        # self.slices.sort(key = lambda n: float(n.split(".")[0]))
        #    self.slices = self.slices[::40]
        # else:
        #    self.slices = self.slices

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(np.load(os.path.join(self.base_path, self.slices[idx])))
        return x.unsqueeze(0)
