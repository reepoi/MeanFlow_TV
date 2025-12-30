import itertools

import torch
from torch.utils.data import IterableDataset
import numpy as np

import conf.dataset


class Dataset(IterableDataset):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__()
        self._generation_batch = 1024
        self.cfg = cfg
        self.rng = rng

    def __iter__(self):
        while True:
            generation_batch = self.generate(self._generation_batch)
            yield from generation_batch

    def generate(self, size):
        raise NotImplementedError()


class Gaussian(Dataset):
    def generate(self, size):
        return torch.tensor(
            self.rng.normal(scale=self.cfg.std, size=(size, self.cfg.dim)),
            dtype=torch.float32,
        )


class Crescent(Dataset):
    def generate(self, size):
        # Calculate the area ratio to estimate required samples
        outer_area = np.pi * self.cfg.outer_radius**2
        inner_area = np.pi * self.cfg.inner_radius**2
        crescent_area = outer_area - inner_area

        # Estimate required samples with 20% buffer
        n_samples = int(size * (outer_area / crescent_area) * 1.2)
        n_samples = max(n_samples, size)  # Ensure we generate at least num_points

        # Generate points in the outer circle
        theta = 2 * np.pi * torch.tensor(
            self.rng.uniform(size=size),
            dtype=torch.float32,
        )
        radius = self.cfg.outer_radius * torch.tensor(
            self.rng.uniform(size=size),
            dtype=torch.float32,
        )

        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)

        # Filter points that are outside the inner circle
        mask = (x - self.cfg.inner_center)**2 + y**2 > self.cfg.inner_radius**2
        points = torch.stack((x[mask], y[mask]), dim=1)

        # If we didn't get enough points, recursively generate more
        while points.shape[0] < size:
            additional_points = self.generate(size - points.shape[0])
            points = torch.cat((points, additional_points), dim=0)

        return points[:size].to(dtype=torch.float32)


def get_dataset(cfg, rng):
    match cfg:
        case conf.dataset.Gaussian():
            return Gaussian(cfg, rng)
        case conf.dataset.Crescent():
            return Crescent(cfg, rng)
        case _:
            raise ValueError(f'Unknown dataset: {cfg}')
