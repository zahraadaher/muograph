import torch
from torch import Tensor
from typing import Optional, Dict
from muograph.plotting.plotting import plot_hist
from muograph.plotting.voxel import VoxelPlotting

r"""
Provides classes for computing regression error metrics.
"""


class RegressionErrorMetrics:
    r"""
    Class for computing regression error metrics from voxelized gronud truth and predictions.
    ground truth and predictions must have the same shape.
    It is advised to normalize the data before computing metrics.
    """

    def __init__(self, ground_truth: Tensor, predictions: Tensor, mask: Optional[Tensor] = None) -> None:
        """
        Initializes the RegressionErrorMetrics class.
        Args:
            ground_truth (torch.Tensor): The ground truth tensor of shape (Nx, Ny, Nz).
            predictions (torch.Tensor): The predicted tensor of the same shape as ground_truth.
        """
        assert (
            ground_truth.shape == predictions.shape
        ), f"Ground truth and predictions must have the same shape, but have {ground_truth.shape} and {predictions.shape}."

        self.ground_truth = ground_truth[mask]
        self.predictions = predictions[mask]

        self.errors = self.ground_truth - self.predictions

    def normalize(self, data: Tensor) -> Tensor:
        """
        Normalizes the data to have a range of [0, 1].
        Args:
            data (torch.Tensor): Input tensor to normalize.
        Returns:
            torch.Tensor: Normalized tensor with range [0, 1].
        """
        data_min = torch.min(data)
        data_max = torch.max(data)
        return (data - data_min) / (data_max - data_min + 1e-8)  # Adding epsilon to avoid division by zero

    def normalize_data(self) -> None:
        """
        Normalizes both ground truth and predictions to have a range of [0, 1].
        Updates self.ground_truth and self.predictions.
        """
        self.ground_truth = self.normalize(self.ground_truth)
        self.predictions = self.normalize(self.predictions)
        self.errors = self.ground_truth - self.predictions

    @property
    def mae(self) -> float:
        """Computes the Mean Absolute Error (MAE)."""
        mae = torch.mean(torch.abs(self.errors))
        return mae.detach().cpu().item()

    @property
    def mse(self) -> float:
        """Computes the Mean Squared Error (MSE)."""
        mse = torch.mean(self.errors**2)
        return mse.detach().cpu().item()

    @property
    def rmse(self) -> float:
        """Computes the Root Mean Squared Error (RMSE)."""
        mse = self.mse
        rmse = torch.sqrt(torch.tensor(mse))
        return rmse.detach().cpu().item()

    @property
    def r_squared(self) -> float:
        """Computes the R-squared (coefficient of determination) metric."""
        total_variance = torch.var(self.ground_truth)
        unexplained_variance = torch.var(self.errors)
        r2 = 1 - (unexplained_variance / total_variance)
        return r2.detach().cpu().item()

    def summary(self, normalize: bool = False) -> Dict[str, float]:
        """
        Generates a summary of all metrics.
        Args:
            normalize (bool): If True, normalizes the data before computing metrics.
        Returns:
            dict: A dictionary of error metrics.
        """
        if normalize:
            self.normalize_data()
        return {"MAE": self.mae, "MSE": self.mse, "RMSE": self.rmse, "R^2": self.r_squared}

    def plot_preds(self, logx: bool = False, logy: bool = False, figname: Optional[str] = None) -> None:
        plot_hist(data_1D=self.predictions.ravel(), logx=logx, logy=logy, xlabel="Predictions", figname=figname)

    def plot_gt(self, logx: bool = False, logy: bool = False, figname: Optional[str] = None) -> None:
        plot_hist(data_1D=self.ground_truth.ravel(), logx=logx, logy=logy, xlabel="Ground Truth", figname=figname)

    def plot_preds_gt_1D(self, dim: int = 0, title: Optional[str] = None) -> None:
        VoxelPlotting.plot_3D_to_1D([self.predictions, self.ground_truth], data_labels=["Predictions", "Ground Truth"], dim=dim, title=title)
