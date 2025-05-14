import torch
from typing import Tuple
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from fastprogress import progress_bar
from muograph.plotting.voxel import VoxelPlotting
from muograph.volume.volume import Volume

import matplotlib.pyplot as plt
import numpy as np


from matplotlib.colors import LogNorm, Normalize

# from matplotlib.gridspec import GridSpec


# from matplotlib.animation import FuncAnimation


# import matplotlib.animation as animation


# from muograph.tracking.tracking_em import Tracking_EM
from muograph.reconstruction.tracking_em_test import TrackingEM  # Aqui es donde yo he ido trabajando y donde tengo M y W


class EM(VoxelPlotting):
    def __init__(
        self,
        voi: Volume,
        tracks: TrackingEM,
        # data: pandas.DataFrame,
        em_iter: int = 1,
        init_lrad: float = 0.5,
    ) -> None:
        VoxelPlotting.__init__(self, voi)
        self.voi = voi
        self.voxel_edges = voi.voxel_edges
        self.Nvox_Z = voi.n_vox_xyz[2]
        # self.data = tracks.data # solo necesito calcualr el momento, no hace falta que defina el dataframe data
        self.mom = tracks.momentum
        self.em_iter = em_iter
        self.init_lrad = init_lrad
        # self.intersection_coordinates = tracks.intersection_coordinates
        # self.triggered_voxels = tracks.triggered_voxels
        # self.Hit = tracks.Hit
        # self.W, self.M, self.Path_Length, self.T2, self.Hit = (
        #     tracks.W,
        #     tracks.M,
        #     tracks.Path_Length,
        #     tracks.T2,
        #     tracks.Hit,
        # )
        # self.indices = tracks.indices # me servian para ahcer un masking, pero realmente no me hace falta
        self.intersection_coordinates = tracks.intersection_coordinates
        self.triggered_voxels = tracks.triggered_voxels
        self.W, self.M, self.Path_Length, self.T2 = (
            tracks.W,
            tracks.M_voxels,
            tracks.L,
            tracks.T,
        )
        self.Hit = tracks.Hit
        self.Dx, self.Dy = tracks.Dx, tracks.Dy

        num_empty_hits = sum([self.Hit[i].sum() == 0 for i in range(len(self.triggered_voxels))])
        print(f"Eventos sin impactos: {num_empty_hits}/{len(self.triggered_voxels)} ({100 * num_empty_hits / len(self.triggered_voxels):.2f}%)")

        # print("M stats:")
        # print("  M shape:", self.M.shape)
        # print("  M min:", torch.min(self.M).item())
        # print("  M max:", torch.max(self.M).item())
        # print("  M mean:", torch.mean(self.M).item())
        # print("  M std:", torch.std(self.M).item())
        # total_voxels = voi.n_vox_xyz[0]*voi.n_vox_xyz[1]*voi.n_vox_xyz[2]
        # print("  Total number of voxels:", total_voxels)
        # print("  M non-zero elements:", torch.count_nonzero(self.M).item())
        # zero_elements = total_voxels-torch.count_nonzero(self.M).item()
        # print("  M zero elements:", zero_elements)

        # self.histograma_M(log_y=False)

        self.pr, self._lambda_ = self.compute_init_scatter_density()
        # print("Momento pr: ", self.pr.size())
        # print("Lambda: ", self._lambda_.size())
        self.rad_length, self.scattering_density = self.em_reconstruction()

    def histograma_M(self, log_y: bool = True) -> None:
        """
        Muestra un histograma de los valores del tensor M, descartando ceros.
        Parámetros:
            log_y (bool): Si es True, usa escala logarítmica en el eje y.
        """
        M_np = self.M.cpu().numpy().flatten()
        # M_np_nonzero = M_np[M_np > 0]  # Filtra ceros

        plt.figure(figsize=(8, 6))
        # plt.hist(M_np_nonzero, bins=100, color='blue', alpha=0.7)
        plt.hist(M_np, bins=100, color="blue", alpha=0.7)
        plt.title("M histogram")
        plt.xlabel("M value")
        plt.ylabel("- (log)" if log_y else "-")

        if log_y:
            plt.yscale("log")

        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

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
        # pr = p0 / (self.data["mom"][self.indices])  # mom has to be in GeV
        pr = p0 / (self.mom)  # mom has to be in GeV

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
        # Es como un objeto (parecido a una lista de elementos) donde cada elemento es una tupla (tensor,) donde el tensor es de un único valor (del 0 al n_events-1: tensor(0),tensor(1),tensor(2)...)
        print(dataset[0][0], dataset[1])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # <-- NEW LINE
        # DataLoader(...) crea un iterador que permite recorrer el dataset en batches del tamaño especificado por batch_size.
        # shuffle=False significa que los datos se entregarán en orden secuencial (0, 1, 2, ...), sin mezclar.

        for itr in progress_bar(
            range(0, self.em_iter - 1)
        ):  # es simplemente para mostrar la barra de progreso en el notebook cuando se ejecuta, de esta manera vemos cuanto le queda
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
                # print('Batch: ', batch)

                for i in batch_indices:  # <-- UPDATED TO USE BATCHED DATA
                    print("\t i: ", i)
                    # el ultimo valor que tomara sera 9477
                    # por loque entiendo, es como si estuvieramos tomando cada muon (i-muon)

                    # pr_i = self.pr[self.indices[i]]
                    pr_i = self.pr[i]
                    tDx_i = torch.transpose(self.Dx[i], 0, -1)  # DiT
                    tDy_i = torch.transpose(self.Dy[i], 0, -1)

                    w_h[i, :, :, :, :, :] = self.W[i, :, :, :, :, :] * lambda_itr[:, :, :, None, None]
                    w_h_sum[i, :, :] = torch.sum(w_h[i, :, :, :, :, :], (0, 1, 2))

                    sigma_D[i, :, :] = (pr_i**2) * w_h_sum[i, :, :]
                    det_sigma_D = torch.det(sigma_D[i, :, :])

                    if det_sigma_D != 0:
                        # print('det_sigma_D != 0')
                        sigma_D_inv = torch.linalg.inv(sigma_D[i, :, :])
                        xx, yy, zz = torch.meshgrid(torch.arange(Ni), torch.arange(Nj), torch.arange(Nk))
                        mask = self.Hit[i].bool()
                        # mask = self.M.bool()
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

                        # # comprobación de matrices y calculos:
                        # # Solo para unos pocos eventos
                        # if i in range(0, 10):  # Puedes cambiar estos valores
                        #     print(f"\n[Iter {itr}] Muon {i}:")
                        #     print(f"  pr_i = {pr_i.item():.3f}")
                        #     print(f"  det(sigma_D) = {det_sigma_D.item():.3e}")
                        #     print(f"  lambda_j stats: min={lambda_j.min():.2e}, max={lambda_j.max():.2e}, mean={lambda_j.mean():.2e}")

                        #     diff = mtr_x_4 - mtr_5
                        #     print(f"  (mtr_x_4 - mtr_5) stats: min={diff.min():.2e}, max={diff.max():.2e}, mean={diff.mean():.2e}")

                        #     # Verificar tamaños
                        #     print(f"  Shapes: mtr_x_4={mtr_x_4.shape}, mtr_5={mtr_5.shape}, lambda_j={lambda_j.shape}")

                        # Si descomento estas lineas el codigo se peta y no se ejecuta
                        # if torch.isnan(Sx).any() or torch.isnan(Sy).any():
                        #     print(f"NaN en Sx o Sy en evento {i}")

                        S[i, mask] = (Sx[i, mask] + Sy[i, mask]) / 2

            # scatter_density[itr + 1, :, :, :] = torch.sum(S, dim=0) / (2 * self.M)

            # denom = 2 * self.M
            # denom[denom == 0] = 1e-12  # para evitar división por cero
            # scatter_density[itr + 1, :, :, :] = torch.sum(S, dim=0) / denom

            mask = self.M != 0
            scatter_density[itr + 1][mask] = (torch.sum(S, dim=0) / (2 * self.M))[mask]

        rad_len = (15e-3 / p_0) ** 2 / scatter_density

        return rad_len, scatter_density

    def plot_scattering_density_slices(self, scatter_density_all: torch.Tensor, iteration: int = -1, log_scale: bool = False) -> None:
        """
        Plots 8 slices along the z-axis from a specific iteration of the 3D scattering density tensor.

        Parameters:
        - scatter_density_all: Tensor of shape (N, Nx, Ny, Nz), all iterations of scattering densities.
        - iteration: Index of the iteration to plot (default: -1, the last one).
        - log_scale: If True, use logarithmic color scale.
        """
        voxel_size_mm = self.voi.vox_width
        z_start = self.voi.xyz_min[2]

        if iteration >= scatter_density_all.shape[0] or iteration < -scatter_density_all.shape[0]:
            raise ValueError(f"Invalid iteration index {iteration}. Valid range: [-{scatter_density_all.shape[0]}, {scatter_density_all.shape[0]-1}]")

        scatter_density = scatter_density_all[iteration]
        scatter_density_np = scatter_density.cpu().numpy()
        Nk = scatter_density_np.shape[2]

        # last_density_np = scatter_density_all[-1].cpu().numpy()
        # vmin = np.min(last_density_np[last_density_np > 0]) if log_scale else np.min(last_density_np)
        # vmax = np.max(last_density_np)
        # if vmin <= 0 or vmin == vmax:
        #     vmin, vmax = 1e-6, 1e-2

        vmin = np.min(scatter_density_np)
        vmax = np.max(scatter_density_np)

        norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)

        z_indices = np.linspace(0, Nk - 1, 8, dtype=int)

        fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

        for ax, k in zip(axs.flat, z_indices):
            img = ax.imshow(
                scatter_density_np[:, :, k].T,
                origin="lower",
                cmap="jet",
                norm=norm,
                extent=[self.voi.xyz_min[0], self.voi.xyz_max[0], self.voi.xyz_min[1], self.voi.xyz_max[1]],
            )
            z_mm_low = z_start + k * voxel_size_mm
            z_mm_high = z_mm_low + voxel_size_mm
            ax.set_title(f"z ∈ [{z_mm_low:.0f}, {z_mm_high:.0f}] mm")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        fig.suptitle(f"Scattering density - Iteration {iteration % scatter_density_all.shape[0]}\nVoxel size = {voxel_size_mm} mm", fontsize=16)
        cbar = fig.colorbar(img, ax=axs.ravel().tolist(), shrink=0.95)
        cbar.set_label("Scattering density", rotation=270, labelpad=20)
        plt.show()

    # # Prueba de animación
    # def animate_scattering_density_slices(
    #     self, scatter_density_all: torch.Tensor, log_scale: bool = False, save_path: str = None, interval: int = 500  # ms per frame
    # ) -> None:
    #     voxel_size_mm = self.voi.vox_width
    #     z_start = self.voi.xyz_min[2]

    #     scatter_density_np_all = scatter_density_all.cpu().numpy()
    #     N_iter = scatter_density_np_all.shape[0]
    #     Nk = scatter_density_np_all.shape[3]
    #     z_indices = np.linspace(0, Nk - 1, 8, dtype=int)

    #     # Common color normalization
    #     last_density_np = scatter_density_np_all[-1]
    #     vmin = np.min(last_density_np[last_density_np > 0]) if log_scale else np.min(last_density_np)
    #     vmax = np.max(last_density_np)
    #     norm = LogNorm(vmin=vmin, vmax=vmax) if log_scale else Normalize(vmin=vmin, vmax=vmax)

    #     fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    #     ims = []
    #     titles = []

    #     for ax in axs.flat:
    #         im = ax.imshow(
    #             np.zeros_like(scatter_density_np_all[0, :, :, 0].T),
    #             origin="lower",
    #             cmap="jet",
    #             norm=norm,
    #             extent=[
    #                 self.voi.xyz_min[0],
    #                 self.voi.xyz_max[0],
    #                 self.voi.xyz_min[1],
    #                 self.voi.xyz_max[1],
    #             ],
    #         )
    #         ims.append(im)
    #         titles.append(ax.set_title(""))
    #         ax.set_xlabel("x [mm]")
    #         ax.set_ylabel("y [mm]")

    #     suptitle = fig.suptitle("", fontsize=16)
    #     cbar = fig.colorbar(ims[0], ax=axs.ravel().tolist(), shrink=0.95)
    #     cbar.set_label("Scattering density", rotation=270, labelpad=20)

    #     def update(frame) -> None:
    #         scatter_density_np = scatter_density_np_all[frame]
    #         for im, title, k, ax in zip(ims, titles, z_indices, axs.flat):
    #             im.set_data(scatter_density_np[:, :, k].T)
    #             z_mm_low = z_start + k * voxel_size_mm
    #             z_mm_high = z_mm_low + voxel_size_mm
    #             title.set_text(f"z ∈ [{z_mm_low:.0f}, {z_mm_high:.0f}] mm")
    #         suptitle.set_text(f"Scattering density - Iteration {frame}\nVoxel size = {voxel_size_mm} mm")
    #         return ims + titles + [suptitle]

    #     anim = FuncAnimation(fig, update, frames=N_iter, interval=interval, blit=False)

    #     if save_path:
    #         anim.save(save_path, writer="ffmpeg", dpi=200)
    #     else:
    #         plt.show()

    def sencilla_plot_scattering_density_slices(self, scatter_density_all: torch.Tensor, iteration: int = -1) -> None:
        """
        Plots 8 slices along the z-axis from a specific iteration of the 3D scattering density tensor.

        Parameters:
        - scatter_density_all: Tensor of shape (N, Nx, Ny, Nz), all iterations of scattering densities.
        - iteration: Index of the iteration to plot (default: -1, the last one).
        """
        voxel_size_mm = self.voi.vox_width
        z_start = self.voi.xyz_min[2]

        # Validación
        if iteration >= scatter_density_all.shape[0] or iteration < -scatter_density_all.shape[0]:
            raise ValueError(f"Invalid iteration index {iteration}. Valid range: [-{scatter_density_all.shape[0]}, {scatter_density_all.shape[0]-1}]")

        # Tensor a mostrar
        scatter_density = scatter_density_all[iteration]
        scatter_density_np = scatter_density.cpu().numpy()
        Nk = scatter_density_np.shape[2]

        # Para mantener escala fija basada en la última iteración
        last_density_np = scatter_density_all[-1].cpu().numpy()
        vmin = np.min(last_density_np)
        vmax = np.max(last_density_np)
        if vmin == vmax:
            vmin, vmax = 1e-6, 1e-5  # Escala por defecto si todos los valores son iguales

        # Slices en Z
        z_indices = np.linspace(0, Nk - 1, 8, dtype=int)

        fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

        for ax, k in zip(axs.flat, z_indices):
            img = ax.imshow(
                scatter_density_np[:, :, k].T,
                origin="lower",
                cmap="jet",
                vmin=vmin,
                vmax=vmax,
                extent=[self.voi.xyz_min[0], self.voi.xyz_max[0], self.voi.xyz_min[1], self.voi.xyz_max[1]],
            )
            z_mm_low = z_start + k * voxel_size_mm
            z_mm_high = z_mm_low + voxel_size_mm
            ax.set_title(f"z ∈ [{z_mm_low:.0f}, {z_mm_high:.0f}] mm")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        fig.suptitle(f"Scattering density - Iteration {iteration % scatter_density_all.shape[0]}\nVoxel size = {voxel_size_mm} mm", fontsize=16)
        cbar = fig.colorbar(img, ax=axs.ravel().tolist(), shrink=0.95)
        cbar.set_label("Scattering density", rotation=270, labelpad=20)
        plt.show()

    def NO_plot_scattering_density_slices(self, scatter_density: torch.Tensor) -> None:
        """
        Plots 8 slices along the z-axis from the 3D scattering density tensor using logarithmic color scale.
        """
        voxel_size_mm = self.voi.vox_width
        z_start = self.voi.xyz_min[2]

        scatter_density_np = scatter_density.cpu().numpy()
        Nk = scatter_density_np.shape[2]

        # Evita log(0): asegúrate de tener valores positivos no nulos
        scatter_density_np = np.clip(scatter_density_np, a_min=1e-6, a_max=None)

        z_indices = np.linspace(0, Nk - 1, 8, dtype=int)

        fig, axs = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)

        vmin = scatter_density_np.min()
        vmax = scatter_density_np.max()

        for ax, k in zip(axs.flat, z_indices):
            img = ax.imshow(
                scatter_density_np[:, :, k].T,
                origin="lower",
                cmap="jet",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                extent=[self.voi.xyz_min[0], self.voi.xyz_max[0], self.voi.xyz_min[1], self.voi.xyz_max[1]],
            )
            z_mm_low = z_start + k * voxel_size_mm
            z_mm_high = z_mm_low + voxel_size_mm
            ax.set_title(f"z ∈ [{z_mm_low:.0f}, {z_mm_high:.0f}] mm")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")

        fig.suptitle("Scattering density predictions (log scale)\nvoxel size = {} mm".format(voxel_size_mm), fontsize=16)
        cbar = fig.colorbar(img, ax=axs.ravel().tolist(), shrink=0.95)
        cbar.set_label("Scattering density (log scale)", rotation=270, labelpad=20)
        plt.show()
