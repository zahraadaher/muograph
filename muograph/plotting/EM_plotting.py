import matplotlib
import matplotlib.pyplot as plt
from torch import Tensor
from typing import List, Optional
import numpy as np
from muograph.plotting.params import d_unit
from muograph.volume.volume import Volume
from muograph.tracking.tracking_em import Tracking_EM


def plot_EM_tracks(voi: Volume, plot_name: Optional[str] = None, event: int = 0, tracking: Tracking_EM = None, poca: bool = False) -> None:
    r"""
    Plot muon tracks inside the voxelized volume of interest.

    Args:
        - voi : volume of interest,
        - event (int) : event number
        - tracking : a Tracking_EM class holding the variables of muon tracks inside the volume, as calculated with the EM algorithm
        - poca (bool) : if False, tracks are plotted for the straight line approximation. If True, the muon tracks consists of two connected tracks: an inwards
          track connecting the point of entry to the volume to the PoCA point, and an outwards track connecting the PoCA to the point at which the muon exits the volume
    """

    def plot_muon_tracks(
        ax: matplotlib.axes.Axes, dim: int, indices: List, points: Tensor, label_prefix: Optional[str] = "", color: Optional[str] = "red"
    ) -> None:
        r"""
        Helper function to plot muon tracks
        """
        if poca is False:
            ax.plot(points[dim, :, indices], points[2, :, indices], color=color, label=label_prefix + "track")
        else:
            ax.plot(points[indices, :, dim], points[indices, :, 2], color=color, label=label_prefix + "track")

    xlabels = ["X", "Y"]
    ylabel = "Z"

    fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
    fig.suptitle(
        "Muon tracks for event = {}".format(event),
        fontweight="bold",
        fontsize=15,
    )

    N_voxels = voi.n_vox_xyz
    xyz_max = voi.xyz_max
    xyz_min = voi.xyz_min

    n_bins_x = N_voxels[0]
    n_bins_y = N_voxels[1]
    n_bins_z = N_voxels[2]
    x_min, x_max = xyz_min[0], xyz_max[0]
    y_min, y_max = xyz_min[1], xyz_max[1]
    z_min, z_max = xyz_min[2], xyz_max[2]

    x_edges = np.linspace(x_min, x_max, n_bins_x + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_y + 1)
    z_edges = np.linspace(z_min, z_max, n_bins_z + 1)

    for dim, edges in zip([0, 1], [x_edges, y_edges]):
        for _ in edges:
            ax[dim].axvline(x=_, color="gray", linewidth=1, linestyle="-")
        for _ in z_edges:
            ax[dim].axhline(y=_, color="gray", linewidth=1, linestyle="-")

        ax[dim].set_xlabel(xlabels[dim] + f" [{d_unit}]")
        ax[dim].set_ylabel(ylabel + f" [{d_unit}]")

        if poca:
            plot_muon_tracks(ax[dim], dim, tracking.indices_in[event], tracking.points_in_POCA, label_prefix="Incoming ", color="red")
            plot_muon_tracks(ax[dim], dim, tracking.indices_out[event], tracking.points_out_POCA, label_prefix="Outgoing ", color="blue")
        else:
            if isinstance(tracking.tracks, Tensor):
                plot_muon_tracks(ax[dim], dim, tracking.indices[event], tracking.tracks)
            else:
                raise TypeError(f"Expected a Tensor, but got {type(tracking.tracks)}.")

        for x, y, z in tracking.triggered_voxels[event]:
            ax[dim].scatter(voi.voxel_centers[x, y, z, dim], voi.voxel_centers[x, y, z, 2], color="red", marker="o", s=50, alpha=0.7, label="triggered voxel")

    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()
