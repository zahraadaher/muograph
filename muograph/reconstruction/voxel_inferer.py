from typing import Optional
import torch
from torch import Tensor

from muograph.volume.volume import Volume
from muograph.tracking.tracking import TrackingMST
from muograph.plotting.voxel import VoxelPlotting


r"""
Provides class for voxel-wise scattering density predictions.
"""


class AbsVoxelInferer(
    VoxelPlotting,
):
    r"""
    Class used for handling the computation and plotting of voxel-wise scattering density predictions
    """

    _xyz_voxel_pred: Optional[Tensor] = None  # (Nx, Ny, Nz)
    _recompute_preds = True

    def __init__(self, voi: Volume, tracking: TrackingMST) -> None:
        r"""
        Initializes the AbsVoxelInferer object with an instance of the TrackingMST class and Volume class.

        Args:
            - voi (Volume) Instance of the Volume class. The BCA algo. relying on voxelized
            volume, `voi` has to be provided.
            - tracking (Optional[TrackingMST]) Instance of the TrackingMST class.
        """

        VoxelPlotting.__init__(self, voi=voi)
        self.tracks = tracking

    def get_xyz_voxel_pred(self) -> Tensor:
        r"""
        Computes the scattering density predictions per voxel.

        Returns:
            vox_density_pred (Tensor): voxelwise density predictions
        """

        pass

    @property
    def xyz_voxel_pred(self) -> Tensor:
        r"""
        The scattering density predictions.
        """
        if (self._xyz_voxel_pred is None) | (self._recompute_preds):
            self._xyz_voxel_pred = self.get_xyz_voxel_pred()
        return self._xyz_voxel_pred

    @property
    def xyz_voxel_pred_norm(self) -> Tensor:
        r"""
        The normalized scattering density predictions.
        """
        return (self.xyz_voxel_pred.float() - torch.min(self.xyz_voxel_pred)) / (torch.max(self.xyz_voxel_pred.float()) - torch.min(self.xyz_voxel_pred))
