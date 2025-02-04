import torch
import math
from copy import deepcopy
from torch import Tensor
import pandas as pd
from typing import Optional, Dict, Tuple
from muograph.plotting.plotting import plot_hist
from muograph.plotting.voxel import VoxelPlotting
from muograph.reconstruction.poca import POCA

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


class PocaErrorMetrics:
    """Class for comparing POCA objects and computing error metrics based on POCA points."""

    _distances: Optional[Tensor] = None  # Cached distances tensor (N,)
    _masks: Optional[Tuple[Tensor, Tensor]] = None  # Cached masks tuple (N,)

    def __init__(self, poca_ref: "POCA", poca: "POCA", output_dir: Optional[str] = None, label: Optional[str] = None):
        """
        Initializes the PocaErrorMetrics class.

        Args:
            poca_ref (POCA): Reference POCA instance.
            poca (POCA): POCA instance to compare against the reference.
            output_dir (Optional[str]): Directory to save outputs.
            label (Optional[str]): Label for saving outputs.
        """
        if not isinstance(poca_ref, POCA) or not isinstance(poca, POCA):
            raise TypeError("Both poca_ref and poca must be instances of the POCA class.")

        self.poca_ref = poca_ref
        self.poca = poca
        self.output_dir = output_dir
        self.label = label

        self.n_mu_ref = deepcopy(poca_ref.n_mu)
        self.n_mu = deepcopy(poca.n_mu)
        self.n_mu_lost = self.n_mu - self.n_mu_ref

        self._filter_events()

    @staticmethod
    def get_compatible_event_masks(poca1: "POCA", poca2: "POCA") -> Tuple[Tensor, Tensor]:
        """
        Finds the compatible events between two POCA objects.

        Args:
            poca1 (POCA): First POCA instance.
            poca2 (POCA): Second POCA instance.

        Returns:
            Tuple[Tensor, Tensor]: Tuple of masks indicating compatible events.
        """
        cross_mask = poca1.full_mask & poca2.full_mask

        def compute_mask(poca: "POCA") -> Tensor:
            valid_indices = torch.where(poca.parallel_mask)[0][poca.mask_in_voi]
            cross_indices = torch.where(cross_mask)[0]
            return torch.isin(valid_indices, cross_indices)

        return compute_mask(poca1), compute_mask(poca2)

    @staticmethod
    def compute_distances(points_1: Tensor, points_2: Tensor) -> Tensor:
        """
        Computes distances between corresponding 3D points from two tensors.

        Args:
            points_1 (Tensor): Tensor of points (N, 3).
            points_2 (Tensor): Tensor of points (N, 3).

        Returns:
            Tensor: Tensor of distances (N,).
        """
        if points_1.shape != points_2.shape or points_1.shape[1] != 3:
            raise ValueError("Input tensors must have shape (N, 3) and be of the same size.")
        return torch.norm(points_1 - points_2, dim=1)

    def _filter_events(self) -> None:
        """Filters events in both POCA objects to retain only compatible events."""
        self.poca_ref._filter_pocas(self.masks[0])
        self.poca_ref.tracks._filter_muons(self.masks[0])
        self.poca._filter_pocas(self.masks[1])
        self.poca.tracks._filter_muons(self.masks[1])

    def plot_distance(self, mask: Optional[Tensor] = None, figname: Optional[str] = None, title: Optional[str] = None) -> None:
        """
        Plots the distribution of distances between the POCA points of two POCA objects.

        Args:
            mask (Optional[Tensor]): Mask to filter distances for plotting.
            figname (Optional[str]): Filename to save the plot.
            title (Optional[str]): Plot title.
        """
        mask = mask if mask is not None else torch.ones_like(self.distances, dtype=torch.bool)
        figname = figname or (self.output_dir and f"{self.output_dir}/distance_distribution")
        title = title or f"Distance between POCA points - {mask.sum().item():,d} events"

        plot_hist(data_1D=self.distances[mask], xlabel="Distance [mm]", figname=figname, title=title, logy=True)

    def save(self) -> None:
        """Saves the summary as a CSV file in the specified output directory."""
        filename = (self.output_dir or "") + (self.label or "poca_metric_summary")
        pd.DataFrame(self.summary).to_csv(filename)
        print(f"Poca metric summary saved at {filename}")

    @property
    def distances(self) -> Tensor:
        """Returns the distances between POCA points."""
        if self._distances is None:
            self._distances = self.compute_distances(self.poca_ref.poca_points, self.poca.poca_points)
        return self._distances

    @property
    def masks(self) -> Tuple[Tensor, Tensor]:
        """Returns masks of compatible events."""
        if self._masks is None:
            self._masks = self.get_compatible_event_masks(self.poca_ref, self.poca)
        return self._masks

    @property
    def summary(self) -> Dict[str, float]:
        """Returns a summary of metrics for the POCA comparison."""
        return {
            "angular_res": self.poca.tracks.angular_res_in,
            "d_mean": self.distances.mean().item(),
            "d_mean_1deg": self.distances[self.poca_ref.tracks.dtheta > 1 * math.pi / 180].mean().item(),
            "d_mean_3deg": self.distances[self.poca_ref.tracks.dtheta > 3 * math.pi / 180].mean().item(),
            "d_std": self.distances.std().item(),
            "d_std_1deg": self.distances[self.poca_ref.tracks.dtheta > 1 * math.pi / 180].std().item(),
            "d_std_3deg": self.distances[self.poca_ref.tracks.dtheta > 3 * math.pi / 180].std().item(),
            "n_mu_ref": self.n_mu_ref,
            "n_mu": self.n_mu,
            "n_mu_lost": self.n_mu_lost,
            "n_mu_shared": self.poca_ref.n_mu,
        }
