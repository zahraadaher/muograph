from typing import Optional
from torch import Tensor
from torch.nn.functional import normalize

from muograph.volume.volume import Volume
from muograph.tracking.tracking import TrackingMST
from muograph.plotting.voxel import VoxelPlotting


class AbsVoxelInferer(
    VoxelPlotting,
):
    _xyz_voxel_pred: Optional[Tensor] = None  # (Nx, Ny, Nz)
    _recompute_preds = True

    def __init__(self, voi: Volume, tracking: TrackingMST) -> None:
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
        return normalize(self.xyz_voxel_pred.float())
