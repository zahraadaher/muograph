import torch
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from fastprogress import progress_bar
from muograph.plotting.voxel import VoxelPlotting
from muograph.volume.volume import Volume
from muograph.tracking.tracking_em import Tracking_EM


class EM(VoxelPlotting):
    def __init__(
        self,
        voi: Volume,
        tracks: Tracking_EM,
        # data: pandas.DataFrame,
        em_iter: int = 1,
        init_lrad: float = 0.5,
    ) -> None:
        VoxelPlotting.__init__(self, voi)
        self.voi = voi
        self.voxel_edges = voi.voxel_edges
        self.Nvox_Z = voi.n_vox_xyz[2]
        self.data = tracks.data
        self.em_iter = em_iter
        self.init_lrad = init_lrad
        self.intersection_coordinates = tracks.intersection_coordinates
        self.triggered_voxels = tracks.triggered_voxels
        self.W, self.M, self.Path_Length, self.T2, self.Hit = (
            tracks.W,
            tracks.M,
            tracks.Path_Length,
            tracks.T2,
            tracks.Hit,
        )
        self.indices = tracks.indices
        self.Dx, self.Dy = tracks.Dx, tracks.Dy
        self.pr, self._lambda_ = self.compute_init_scatter_density()
        self.rad_length, self.scattering_density = self.em_reconstruction()

    def compute_init_scatter_density(self) -> Tuple[Tensor, Tensor]:
        """
        Function calculates the initial scatter density based on the input voxelized object.

        Outputs:

            pr: a torch tensor of shape (Ni, Nj, Nk),
                containing the density of the medium that the muon path crosses
            lambda: a torch tensor of shape (Ni, Nj, Nk), containing the attenuation
                    coefficient of the medium that the muon path crosses
        """

        L_rad = torch.full(
            (self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2]),
            self.init_lrad,
        )
        p0 = 5  # GeV
        _lambda_ = ((15e-3 / p0) ** 2) * (1 / L_rad)
        pr = p0 / (self.data["mom"][self.indices])  # mom has to be in GeV

        return pr, _lambda_

    def em_reconstruction(self, batch_size: int = 100) -> Tuple[Tensor, Tensor]:
        """
        Batch version of the EM reconstruction function to process events in smaller chunks.
        """
        n_events = len(self.triggered_voxels)
        Ni, Nj, Nk = self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2]
        scatter_density = torch.zeros(self.em_iter, Ni, Nj, Nk)
        scatter_density[0] = self._lambda_
        p_0 = 5  # GeV

        print("Performing the Expectation and Maximization (EM) steps in batches.")

        # Create a DataLoader for batching the events
        dataset = TensorDataset(torch.arange(n_events))  # <-- NEW LINE
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # <-- NEW LINE

        for itr in progress_bar(range(0, self.em_iter - 1)):
            sigma_D = torch.zeros(n_events, 2, 2)
            w_h = torch.zeros(n_events, Ni, Nj, Nk, 2, 2)
            w_h_sum = torch.zeros(n_events, 2, 2)
            Sx = torch.zeros(n_events, Ni, Nj, Nk)
            Sy = torch.zeros(n_events, Ni, Nj, Nk)
            S = torch.zeros(n_events, Ni, Nj, Nk)
            lambda_itr = scatter_density[itr]
            print(f"Scatter Density at iteration {itr}: {scatter_density[itr]}")

            for batch in dataloader:  # <-- NEW LINE
                batch_indices = batch[0].tolist()  # Extract batch indices  # <-- NEW LINE

                for i in batch_indices:  # <-- UPDATED TO USE BATCHED DATA
                    pr_i = self.pr[self.indices[i]]
                    tDx_i = torch.transpose(self.Dx[i], 0, -1)  # DiT
                    tDy_i = torch.transpose(self.Dy[i], 0, -1)

                    w_h[i, :, :, :, :, :] = self.W[i, :, :, :, :, :] * lambda_itr[:, :, :, None, None]
                    w_h_sum[i, :, :] = torch.sum(w_h[i, :, :, :, :, :], (0, 1, 2))

                    sigma_D[i, :, :] = (pr_i**2) * w_h_sum[i, :, :]
                    det_sigma_D = torch.det(sigma_D[i, :, :])

                    if det_sigma_D != 0:
                        sigma_D_inv = torch.linalg.inv(sigma_D[i, :, :])
                        xx, yy, zz = torch.meshgrid(torch.arange(Ni), torch.arange(Nj), torch.arange(Nk))
                        mask = self.Hit[i].bool()
                        xx = xx[mask]
                        yy = yy[mask]
                        zz = zz[mask]
                        lambda_j = lambda_itr[xx, yy, zz]
                        w = self.W[i, xx, yy, zz, :, :]

                        mtr_x_1 = torch.matmul(tDx_i, sigma_D_inv)
                        mtr_x_2 = torch.matmul(mtr_x_1, w)
                        mtr_x_3 = torch.matmul(mtr_x_2, sigma_D_inv)
                        mtr_x_4 = torch.matmul(mtr_x_3, self.Dx[i])

                        mtr_y_1 = torch.matmul(tDy_i, sigma_D_inv)
                        mtr_y_2 = torch.matmul(mtr_y_1, w)
                        mtr_y_3 = torch.matmul(mtr_y_2, sigma_D_inv)
                        mtr_y_4 = torch.matmul(mtr_y_3, self.Dy[i])

                        mtr_5 = (
                            torch.einsum(
                                "ijk,ikl->ijl",
                                sigma_D_inv.unsqueeze(0).expand(len(xx), 2, 2),
                                w,
                            )
                            .diagonal(dim1=1, dim2=2)
                            .sum(dim=1)
                        )

                        Sx[i, mask] = 2 * lambda_j + (mtr_x_4 - mtr_5) * (pr_i**2) * (lambda_j**2)

                        Sy[i, mask] = 2 * lambda_j + (mtr_y_4 - mtr_5) * (pr_i**2) * (lambda_j**2)

                        S[i, mask] = (Sx[i, mask] + Sy[i, mask]) / 2

            scatter_density[itr + 1, :, :, :] = torch.sum(S, dim=0) / (2 * self.M)

        rad_len = (15e-3 / p_0) ** 2 / scatter_density

        return rad_len, scatter_density
