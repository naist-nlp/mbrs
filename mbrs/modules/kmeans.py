from typing import Tuple

import torch
from torch import Tensor


class Kmeans:
    """k-means clustering implemented in PyTorch.

    Args:
        ncentroids (int): The number of centroids.
        dim (int): The dimension size of centroids.
        metric (Metric): Distance metric function.
        kmeanspp (bool): Use k-means++ for centroid intialization.
    """

    def __init__(self, ncentroids: int, dim: int, kmeanspp: bool = False) -> None:
        self.ncentroids = ncentroids
        self.dim = dim
        self.kmeanspp = kmeanspp

    @property
    def centroids(self) -> Tensor:
        """Returns centroids tensor of shape `(ncentroids, dim)`."""
        return self._centroids

    @centroids.setter
    def centroids(self, centroids: Tensor) -> None:
        """Sets the given tensor as the centroids.

        Args:
            centroids (Tensor): Centroids tensor of shape `(ncentroids, dim)`.
        """
        self._centroids = centroids

    def assign(self, x: Tensor) -> Tensor:
        """Assigns the nearest neighbor centroid ID.

        Args:
            x (torch.Tensor): Assigned vectors of shape `(n, dim)`.

        Returns:
            torch.Tensor: Assigned IDs of shape `(n,)`.
        """
        return torch.cdist(x, self.centroids, p=2).argmin(dim=-1)

    def update(self, x: Tensor, assigns: Tensor) -> Tensor:
        """Updates the centroids.

        Args:
            x (torch.Tensor): Sample vectors of shape `(n, dim)`.
            assigns (torch.Tensor): Assigned centroids of the given input vectors of shape `(n,)`.

        Returns:
            torch.Tensor: New centroid vectors of shape `(ncentroids, dim)`.
        """
        new_centroids = self.centroids
        for k in range(self.ncentroids):
            if (assigns == k).any():
                new_centroids[k] = x[assigns == k].mean(dim=0)
        return new_centroids

    def init_kmeanspp(self, x: Tensor) -> Tensor:
        """Initializes the centroids via k-means++.

        Args:
            x (Tensor): Input vectors of shape `(n, dim)`.

        Returns:
            Tensor: Centroid vectors obtained using k-means++.
        """
        centroids = x[torch.randint(x.size(0), size=(1,)), :]
        for _ in range(self.ncentroids - 1):
            # Nc x N
            sqdists = torch.cdist(centroids, x, p=2) ** 2
            assigns = sqdists.argmin(dim=0, keepdim=True)
            neighbor_sqdists = sqdists.gather(dim=0, index=assigns).squeeze(0)
            weights = neighbor_sqdists / neighbor_sqdists.sum()
            new_centroid = x[torch.multinomial(weights, 1), :]
            centroids = torch.cat([centroids, new_centroid])
        assert list(centroids.shape) == [self.ncentroids, self.dim]
        return centroids

    def train(
        self, x: Tensor, niter: int = 5, seed: int = 0, verbose: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, dim)`.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(ncentroids, dim)`.
            Tensor: Assigend IDs of shape `(n,)`.
        """
        if self.ncentroids == 1:
            self.centroids = x.mean(dim=0)
            return self.centroids, self.assign(x)

        torch.manual_seed(seed)
        if self.kmeanspp:
            self.centroids = self.init_kmeanspp(x)
        else:
            self.centroids = x[
                torch.randperm(x.size(0), device=x.device)[: self.ncentroids]
            ]
        assigns = x.new_full((x.size(0),), fill_value=-1)
        iter_cnt = 0
        for i in range(niter):
            new_assigns = self.assign(x)
            if torch.equal(new_assigns, assigns):
                break
            assigns = new_assigns
            self.centroids = self.update(x, assigns)
            iter_cnt += 1
        if verbose:
            print(f"[Kmeans] #iter: {iter_cnt}")
        return self.centroids, assigns
