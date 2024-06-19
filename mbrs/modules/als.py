from typing import Optional, Tuple

import torch
import torch.linalg as LA
from torch import Tensor

from mbrs import timer


class MatrixFactorizationALS:
    """Alternating least squares (ALS) implemented in PyTorch.

    Args:
        regularization_weight (float): Weight of L2 regularization.
        rank (int): Rank of the factarized matrices.
    """

    def __init__(self, regularization_weight: float = 0.1, rank: int = 8) -> None:
        self.regularization_weight = regularization_weight
        self.rank = rank

    def compute_loss(
        self,
        matrix: Tensor,
        x: Tensor,
        y: Tensor,
        observed_mask: Optional[Tensor] = None,
    ) -> float:
        """Compute the objective loss function.

        Args:
            matrix (Tensor): Target matrix of shape `(N, M)`.
            x (Tensor): Left-side low-rank matrix of shape `(N, r)`.
            y (Tensor): Right-side low-rank matrix of shape `(M, r)`.
            observed_mask (Tensor, optional): Valid indices boolean mask of shape `(N, M)`.

        Returns:
            Tensor: Objective loss.
        """
        mse_loss = ((observed_mask * (matrix - x @ (y.T))) ** 2).sum()
        l2_regularization_loss = x.norm() ** 2 + y.norm() ** 2
        loss = mse_loss + self.regularization_weight * l2_regularization_loss
        return loss.item()

    def factorize(
        self,
        matrix: Tensor,
        observed_mask: Optional[Tensor] = None,
        niter: int = 30,
        tolerance: float = 1e-4,
        seed: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Factorize the given matrix.

        The input matrix of shape `(N, M)` is decomposed into `X @ Y.T`,
        where `X` and `Y` shape `(N, r)` and `(M, r)`, respectively.

        This implementation does not compute the inverse matrix directly in `X = A^-1 @ b`.
        Instead, `AX = b` is solved.

        Args:
            matrix (Tensor): Input matrix of shape `(N, M)`.
            observed_mask (Tensor, optional): Boolean mask of valid indices of shape `(N, M)`.
            niter (int): The number of alternating steps performed.
            tolerance (float): If the difference between the previous and current loss
              is smaller this value, ALS is regarded as converged.
            seed (int): A seed for the random number generator.

        Returns:
            Tensor: Low-rank matrix `X` of shape `(N, r)`.
            Tensor: Low-rank matrix `Y` of shape `(M, r)`.
        """
        rng = torch.Generator(matrix.device)
        rng = rng.manual_seed(seed)

        N, M = matrix.size()
        # Initialization
        X = (
            torch.rand((N, self.rank), generator=rng, device=matrix.device)
            * (N * self.rank) ** -0.5
        )
        Y = (
            torch.rand((M, self.rank), generator=rng, device=matrix.device)
            * (M * self.rank) ** -0.5
        )
        if observed_mask is None:
            observed_mask = matrix.new_ones((N, M))

        prev_loss = float("1e5")
        for _ in range(niter):
            with timer.measure("ALS/iteration"):
                # A: r x r
                # B: r x N
                # Solve A @ x.T = b
                for i in range(N):
                    A = Y.T @ (
                        Y * observed_mask[i, :, None]
                    ) + self.regularization_weight * torch.eye(
                        self.rank, dtype=Y.dtype, device=Y.device
                    )

                    b = Y.T @ matrix[i, :]
                    X[i, :] = LA.solve(A, b)

                # Y_A: r x r
                # Y_B: r x M
                # Solve Y_A @ Y.T = Y_B
                for j in range(M):
                    A = X.T @ (
                        X * observed_mask[:, j, None]
                    ) + self.regularization_weight * torch.eye(
                        self.rank, dtype=X.dtype, device=X.device
                    )
                    b = X.T @ matrix[:, j]
                    Y[j, :] = LA.solve(A, b)

                loss = self.compute_loss(matrix, X, Y, observed_mask=observed_mask)
                if prev_loss - loss <= tolerance:
                    break
                prev_loss = loss
        return X, Y
