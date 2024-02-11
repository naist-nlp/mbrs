from typing import Tuple

import torch
from torch import Generator, Tensor

from mbrs import timer


class Kmeans:
    """k-means clustering implemented in PyTorch.

    Args:
        kmeanspp (bool): Use k-means++ for centroid intialization.
    """

    def __init__(self, kmeanspp: bool = False) -> None:
        self.kmeanspp = kmeanspp

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
            sqdists = torch.cdist(centroids, x, p=2).float() ** 2
            assigns = sqdists.argmin(dim=0, keepdim=True)
            neighbor_sqdists = sqdists.gather(dim=0, index=assigns).squeeze(0)
            weights = neighbor_sqdists / neighbor_sqdists.sum()
            new_centroid = x[torch.multinomial(weights, 1, generator=rng), :]
            centroids = torch.cat([centroids, new_centroid])
        return centroids

    def train(
        self, x: Tensor, ncentroids: int, niter: int = 5, seed: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, dim)`.
            ncentroids (int): Number of centroids.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(ncentroids, dim)`.
            Tensor: Assigend IDs of shape `(n,)`.
        """
        if ncentroids == 1:
            with timer.measure("kmeans/iteration"):
                centroids = x.mean(dim=0, keepdim=True)
            return centroids, self.assign(x, centroids)

        with timer.measure("kmeans/initialize"):
            rng = torch.Generator(x.device)
            rng = rng.manual_seed(seed)
            if self.kmeanspp:
                centroids = self.init_kmeanspp(x, rng, ncentroids)
            else:
                centroids = x[
                    torch.randperm(x.size(0), generator=rng, device=x.device)[
                        :ncentroids
                    ]
                ]

        assigns = x.new_full((x.size(0),), fill_value=-1)
        for i in range(niter):
            with timer.measure("kmeans/iteration"):
                new_assigns = self.assign(x, centroids)
                if torch.equal(new_assigns, assigns):
                    break
                assigns = new_assigns
                for k in range(ncentroids):
                    if (assigns == k).any():
                        centroids[k] = x[assigns == k].mean(dim=0)
        return centroids, assigns
