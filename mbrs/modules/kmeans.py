from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Generator, Tensor

from mbrs import timer


class Kmeans:
    """k-means clustering implemented in PyTorch.

    Args:
        cfg (Kmeans.Config): Configuration for k-means.
    """

    @dataclass
    class Config:
        """Configuration for k-means.

        - ncentroids (int): Number of centroids.
        - niter (int): Number of k-means iteration
        - kmeanspp (bool): Initialize the centroids using k-means++.
        - seed (bool): Random seed.
        """

        ncentroids: int = 8
        niter: int = 5
        kmeanspp: bool = True
        seed: int = 0

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def assign(self, x: Tensor, centroids: Tensor) -> Tensor:
        """Assigns the nearest neighbor centroid ID.

        Args:
            x (torch.Tensor): Assigned vectors of shape `(n, dim)`.
            centroids (torch.Tensor): Centroids tensor of shape  `(ncentroids, dim)`.

        Returns:
            torch.Tensor: Assigned IDs of shape `(n,)`.
        """
        return torch.cdist(x, centroids, p=2).argmin(dim=-1)

    def init_kmeanspp(self, x: Tensor, rng: Generator, ncentroids: int) -> Tensor:
        """Initializes the centroids via k-means++.

        Args:
            x (Tensor): Input vectors of shape `(n, dim)`.
            rng (Generator): Random number generator.
            ncentroids (int): Number of centroids.

        Returns:
            Tensor: Centroid vectors obtained using k-means++.
        """
        centroids = x[
            torch.randint(x.size(0), size=(1,), generator=rng, device=x.device), :
        ]
        for _ in range(ncentroids - 1):
            # Nc x N
            sqdists = torch.cdist(centroids, x, p=2) ** 2
            neighbor_sqdists = sqdists.min(dim=0).values.float().clamp(min=1e-5)
            weights = neighbor_sqdists / neighbor_sqdists.sum()
            new_centroid = x[torch.multinomial(weights, 1, generator=rng), :]
            centroids = torch.cat([centroids, new_centroid])
        return centroids

    def train(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, dim)`.

        Returns:
            Tensor: Centroids tensor of shape `(ncentroids, dim)`.
            Tensor: Assigend IDs of shape `(n,)`.
        """
        if self.cfg.ncentroids == 1:
            with timer.measure("kmeans/iteration"):
                centroids = x.mean(dim=0, keepdim=True)
            return centroids, self.assign(x, centroids)
        elif x.size(0) <= self.cfg.ncentroids:
            return x, torch.arange(x.size(0), device=x.device)

        with timer.measure("kmeans/initialize"):
            rng = torch.Generator(x.device)
            rng = rng.manual_seed(self.cfg.seed)
            if self.cfg.kmeanspp:
                centroids = self.init_kmeanspp(x, rng, self.cfg.ncentroids)
            else:
                centroids = x[
                    torch.randperm(x.size(0), generator=rng, device=x.device)[
                        : self.cfg.ncentroids
                    ]
                ]

        assigns = x.new_full((x.size(0),), fill_value=-1)
        for i in range(self.cfg.niter):
            with timer.measure("kmeans/iteration"):
                new_assigns = self.assign(x, centroids)
                if torch.equal(new_assigns, assigns):
                    break
                assigns = new_assigns
                for k in range(self.cfg.ncentroids):
                    if (assigns == k).any():
                        centroids[k] = x[assigns == k].mean(dim=0)
        return centroids, assigns
