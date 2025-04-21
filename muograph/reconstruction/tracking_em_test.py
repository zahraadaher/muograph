from muograph.plotting.voxel import VoxelPlotting
from muograph.plotting.params import configure_plot_theme, font, tracking_figsize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import List, Tuple, Union, Optional, Dict, Callable
import torch
from torch import Tensor
import numpy as np
import math
from fastprogress import progress_bar
from muograph.volume.volume import Volume
from muograph.reconstruction.poca import POCA
from muograph.tracking.tracking import TrackingMST
from muograph.reconstruction.asr import ASR


N_POINTS_PER_Z_LAYER = 7
YNprint = False  # If True, then some prints will appear, if False then they won't appear


class TrackingEM(VoxelPlotting):
    _xyz_in_out_voi: Optional[Tuple[Tensor, Tensor]] = None
    _triggered_voxels: Optional[List[np.ndarray]] = None
    _triggered_voxels_BETA: Optional[List[Tensor]] = None  # For the new method that calculates the triggered voxels based on the intersections coordinates
    _intersection_coordinates: Optional[List[Tensor]] = None
    _path_length_LT: Optional[List[Tensor]] = None  # The L and T path length
    _path_length_in_out: Optional[Tuple[Tensor, Tensor]] = None

    _xyz_enters_voi: Optional[Tensor] = None  # to keep
    _xyz_exits_voi: Optional[Tensor] = None  # to keep

    _all_poca: Optional[Tensor] = None

    _M_voxels: Optional[Tensor] = None  # is the number of muons hitting the each voxel (Nx, Ny, Nz)

    _W: Optional[Tensor] = None  # weight matrix

    def __init__(self, voi: Volume, tracking: TrackingMST, n_events: int = 1000, batch_size: int = 100, muon_path: str = "poca") -> None:
        # The voxelized volume of inetrest
        self.voi = voi

        # Tracking class
        self.tracking = tracking

        # POCA
        self.poca = POCA(tracking=tracking, voi=voi)

        # EM parameters
        self.n_events = n_events
        self.batch_size = batch_size
        self.muon_path = muon_path

        self.M_voxels = self.set_M_voxels()

    # Ignore this functions
    @staticmethod
    def _find_triggered_voxels(
        voi: Volume,
        sub_vol_indices_min_max: List[List[Tensor]],
        xyz_discrete_in: Tensor,
        xyz_discrete_out: Tensor,
    ) -> List[np.ndarray]:
        """Find the voxels triggered by the muon track"""
        triggered_voxels = []
        n_mu = xyz_discrete_in.size(2)

        print("\nVoxel triggering")
        for event in progress_bar(range(n_mu)):
            event_indices = sub_vol_indices_min_max[event]

            if len(sub_vol_indices_min_max[event]) != 0:
                ix_min, iy_min = event_indices[0][0], event_indices[0][1]
                ix_max, iy_max = event_indices[1][0], event_indices[1][1]

                sub_voi_edges = voi.voxel_edges[ix_min : ix_max + 1, iy_min : iy_max + 1]
                sub_voi_edges = sub_voi_edges[:, :, :, :, None, :].expand(-1, -1, -1, -1, xyz_discrete_out.size()[1], -1)

                xyz_in_event = xyz_discrete_in[:, :, event]
                xyz_out_event = xyz_discrete_out[:, :, event]

                sub_mask_in = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_in_event[0])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_in_event[0])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_in_event[1])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_in_event[1])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_in_event[2])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_in_event[2])
                )

                sub_mask_out = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_out_event[0])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_out_event[0])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_out_event[1])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_out_event[1])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_out_event[2])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_out_event[2])
                )

                vox_list = (sub_mask_in & sub_mask_out).nonzero()[:, :-1].unique(dim=0)
                vox_list[:, 0] += ix_min
                vox_list[:, 1] += iy_min
                triggered_voxels.append(vox_list.detach().cpu().numpy())
            else:
                triggered_voxels.append(np.empty([0, 3]))
        return triggered_voxels

    @staticmethod
    def get_triggered_voxels(
        points_in: Tensor,
        points_out: Tensor,
        voi: Volume,
        theta_xy_in: Tuple[Tensor, Tensor],
        theta_xy_out: Tuple[Tensor, Tensor],
    ) -> List[np.ndarray]:
        xyz_in_out_voi = ASR._compute_xyz_in_out(
            points_in=points_in,
            points_out=points_out,
            voi=voi,
            theta_xy_in=theta_xy_in,
            theta_xy_out=theta_xy_out,
        )

        xyz_discrete_in_out = ASR._compute_discrete_tracks(
            xyz_in_out_voi=xyz_in_out_voi,
            theta_xy_in=theta_xy_in,
            theta_xy_out=theta_xy_out,
            n_points_per_z_layer=N_POINTS_PER_Z_LAYER,
            voi=voi,
        )

        sub_vol_indices_min_max = ASR._find_sub_volume(voi=voi, xyz_in_voi=xyz_in_out_voi[0], xyz_out_voi=xyz_in_out_voi[1])

        return TrackingEM._find_triggered_voxels(
            voi=voi,
            sub_vol_indices_min_max=sub_vol_indices_min_max,
            xyz_discrete_in=xyz_discrete_in_out[0],
            xyz_discrete_out=xyz_discrete_in_out[1],
        )

    @staticmethod
    def compute_intersection_coordinates_all_muons(
        voi: Volume, xyz_enters_voi: Tensor, xyz_exits_voi: Tensor, tracks_in: Tensor, tracks_out: Tensor, all_poca: Tensor
    ) -> List[Tensor]:
        """
        A method that retuns the xyz coordinates of the intersection points of the muon track inside the volume with the faces
          of its triggered voxels inside this volume.

        Args:
            voi (Volume): the voxelized volume of interest
            xyz_enters_voi (Tensor): a tensor with the entry x,y and z coordinates (N_mu, 3)
            xyz_exits_voi (Tensor): a tensor with the exit x,y and z coordinates (N_mu, 3)
            tracks_in (Tensor): the unit direction vector of the entering point (N_mu, 3)
            tracks_out (Tensor): the unit direction vector of the exit point (N_mu, 3)
            all_poca (Tensor): tensor of the xyz positions of all unfiltered poca events (N_mu, 3). If the poca xyz is (0,0,0),
              this coresponds to a filtered event.

        Returns:
            List[Tensor]: a list of tensors, each element represents the xyz coordinates of the intersection points of the muon track inside the volume with the faces
              of its triggered voxels inside this volume (N_mu, )
        """
        # Extract the volume limits to determine the planes' positions (de divisions of the volume into voxels)
        x_min, y_min, z_min = voi.xyz_min
        x_max, y_max, z_max = voi.xyz_max

        # Determine the number of voxels along each axis
        nx, ny, nz = voi.n_vox_xyz
        if YNprint:
            print(f"Number of voxels per axes: nx={nx}, ny={ny}, nz={nz}")

        # Define the coordinate planes along the three axes
        planes_X = torch.tensor(np.arange(x_min, x_max + nx, voi.vox_width))  # (nx)
        planes_Y = torch.tensor(np.arange(y_min, y_max + ny, voi.vox_width))  # (ny)
        planes_Z = torch.tensor(np.arange(z_min, z_max + nz, voi.vox_width))  # (nz)
        if YNprint:
            print(f"Dimension X planes before expand: {planes_X.size()}")
            print(f"Dimension x0 coord before expand: {xyz_enters_voi[:, 0].size()}")
            print(f"Dimension x_in coord before expand: {tracks_in[:, 0].size()}")

        # === Compute intersection points with planes for the entering track === #
        # Expand the coordinate planes for broadcasting with muons
        planes_X_expand = planes_X.unsqueeze(0)  # (1, nx)
        x0_in_expand = xyz_enters_voi[:, 0].unsqueeze(1)  # (1,n_muons)
        x_in_expand = tracks_in[:, 0].unsqueeze(1)  # (1, n_muons)
        if YNprint:
            print(f"Dimension X planes after expand (x coord): {planes_X_expand.size()}")
            print(f"Dimension x0 coord after expand: {x0_in_expand.size()}")
            print(f"Dimension x_in coord after expand: {x_in_expand.size()}")

        # Compute intersections with X = constant planes
        t_coord_X_in = (planes_X_expand - x0_in_expand) / x_in_expand  # (n_muons, nx)
        y_coord_X_in = xyz_enters_voi[:, 1].unsqueeze(1) + t_coord_X_in * tracks_in[:, 1].unsqueeze(1)  # (n_muons, nx)
        z_coord_X_in = xyz_enters_voi[:, 2].unsqueeze(1) + t_coord_X_in * tracks_in[:, 2].unsqueeze(1)  # (n_muons, nx)
        intersection_points_X_in = torch.stack((planes_X_expand.expand_as(y_coord_X_in), y_coord_X_in, z_coord_X_in), dim=-1)  # (n_muons, nx, 3)
        if YNprint:
            print(f"Dimension t coord: {t_coord_X_in.size()}")
            print(f"Dimension y coord: {y_coord_X_in.size()}")
            print(f"Dimension z coord: {z_coord_X_in.size()}")
            print(f"Dimension inters_points_X_in: {intersection_points_X_in.size()}")

        # Compute intersections with Y = constant planes
        planes_Y_expand = planes_Y.unsqueeze(0)  # (1, ny)
        y0_in_expand = xyz_enters_voi[:, 1].unsqueeze(1)  # (1, n_muons)
        y_in_expand = tracks_in[:, 1].unsqueeze(1)  # (1, n_muons)
        t_coord_Y_in = (planes_Y_expand - y0_in_expand) / y_in_expand  # (n_muons, ny)
        x_coord_Y_in = xyz_enters_voi[:, 0].unsqueeze(1) + t_coord_Y_in * tracks_in[:, 0].unsqueeze(1)  # (n_muons, ny)
        z_coord_Y_in = xyz_enters_voi[:, 2].unsqueeze(1) + t_coord_Y_in * tracks_in[:, 2].unsqueeze(1)  # (n_muons, ny)
        intersection_points_Y_in = torch.stack((x_coord_Y_in, planes_Y_expand.expand_as(x_coord_Y_in), z_coord_Y_in), dim=-1)  # (n_muons, ny, 3)

        # Compute intersections with Z = constant planes
        planes_Z_expand = planes_Z.unsqueeze(0)  # (1, nz)
        z0_in_expand = xyz_enters_voi[:, 2].unsqueeze(1)  # (1, n_muons)
        z_in_expand = tracks_in[:, 2].unsqueeze(1)  # (1, n_muons)
        t_coord_Z_in = (planes_Z_expand - z0_in_expand) / z_in_expand  # (n_muons, nz)
        x_coord_Z_in = xyz_enters_voi[:, 0].unsqueeze(1) + t_coord_Z_in * tracks_in[:, 0].unsqueeze(1)  # (n_muons, nz)
        y_coord_Z_in = xyz_enters_voi[:, 1].unsqueeze(1) + t_coord_Z_in * tracks_in[:, 1].unsqueeze(1)  # (n_muons, nz)
        intersection_points_Z_in = torch.stack((x_coord_Z_in, y_coord_Z_in, planes_Z_expand.expand_as(x_coord_Z_in)), dim=-1)  # (n_muons, nz, 3)

        # === Compute intersection points for the exiting track === #
        x0_out_expand = xyz_exits_voi[:, 0].unsqueeze(1)  # (1, n_muons)
        x_out_expand = tracks_out[:, 0].unsqueeze(1)  # (1, n_muons)
        t_coord_X_out = (planes_X_expand - x0_out_expand) / x_out_expand  # (n_muons, nx)
        y_coord_X_out = xyz_exits_voi[:, 1].unsqueeze(1) + t_coord_X_out * tracks_out[:, 1].unsqueeze(1)  # (n_muons, nx)
        z_coord_X_out = xyz_exits_voi[:, 2].unsqueeze(1) + t_coord_X_out * tracks_out[:, 2].unsqueeze(1)  # (n_muons, nx)
        intersection_points_X_out = torch.stack((planes_X_expand.expand_as(y_coord_X_out), y_coord_X_out, z_coord_X_out), dim=-1)  # (n_muons, nx, 3)

        y0_out_expand = xyz_exits_voi[:, 1].unsqueeze(1)  # (1, n_muons)
        y_out_expand = tracks_out[:, 1].unsqueeze(1)  # (1, n_muons)
        t_coord_Y_out = (planes_Y_expand - y0_out_expand) / y_out_expand  # (n_muons, ny)
        x_coord_Y_out = xyz_exits_voi[:, 0].unsqueeze(1) + t_coord_Y_out * tracks_out[:, 0].unsqueeze(1)  # (n_muons, ny)
        z_coord_Y_out = xyz_exits_voi[:, 2].unsqueeze(1) + t_coord_Y_out * tracks_out[:, 2].unsqueeze(1)  # (n_muons, ny)
        intersection_points_Y_out = torch.stack((x_coord_Y_out, planes_Y_expand.expand_as(x_coord_Y_out), z_coord_Y_out), dim=-1)  # (n_muons, ny, 3)

        z0_out_expand = xyz_exits_voi[:, 2].unsqueeze(1)  # (1, n_muons)
        z_out_expand = tracks_out[:, 2].unsqueeze(1)  # (1, n_muons)
        t_coord_Z_out = (planes_Z_expand - z0_out_expand) / z_out_expand  # (n_muons, nz)
        x_coord_Z_out = xyz_exits_voi[:, 0].unsqueeze(1) + t_coord_Z_out * tracks_out[:, 0].unsqueeze(1)  # (n_muons, nz)
        y_coord_Z_out = xyz_exits_voi[:, 1].unsqueeze(1) + t_coord_Z_out * tracks_out[:, 1].unsqueeze(1)  # (n_muons, nz)
        intersection_points_Z_out = torch.stack((x_coord_Z_out, y_coord_Z_out, planes_Z_expand.expand_as(x_coord_Z_out)), dim=-1)  # (n_muons, nz, 3)

        # Combine all intersection points for each track
        total_intersection_points_in = torch.cat(
            (intersection_points_X_in, intersection_points_Y_in, intersection_points_Z_in), dim=1
        )  # (n_muons, M, 3) where M = nx + ny + nz
        total_intersection_points_out = torch.cat((intersection_points_X_out, intersection_points_Y_out, intersection_points_Z_out), dim=1)  # (n_muons, M, 3)
        if YNprint:
            print(f"Dimension total_inters_point_in: {total_intersection_points_in.size()}")

        # Compute masks based on enter, exit and POCA position
        z_inicial = xyz_enters_voi[:, 2].unsqueeze(1)
        z_final = xyz_exits_voi[:, 2].unsqueeze(1)
        z_POCA = all_poca[:, 2].unsqueeze(1)

        mask_in = (total_intersection_points_in[:, :, 2] <= z_inicial) & (total_intersection_points_in[:, :, 2] >= z_POCA)
        mask_out = (total_intersection_points_out[:, :, 2] <= z_POCA) & (total_intersection_points_out[:, :, 2] >= z_final)

        # Apply masks and concatenate intersections for each muon
        final_points = []
        for i in range(len(all_poca)):
            filtered_points_in = total_intersection_points_in[i][mask_in[i]]
            filtered_points_out = total_intersection_points_out[i][mask_out[i]]
            muon_points = torch.cat((filtered_points_in, filtered_points_out), dim=0)
            # I am not quite sure about this sorted thing I did, first I sort by the z coordinate, then by de x and finally by the y
            ordered_muon_points_Z = muon_points[torch.argsort(muon_points[:, 2], descending=True)]
            # ordered_muon_points_ZX = ordered_muon_points_Z[torch.argsort(ordered_muon_points_Z[:, 0], descending=True)]
            # ordered_muon_points_ZXY = ordered_muon_points_ZX[torch.argsort(ordered_muon_points_ZX[:, 1], descending=True)]
            # final_points.append(ordered_muon_points_ZXY)
            final_points.append(ordered_muon_points_Z)  # Ordered just by the Z coordinate
            if i == 0 and YNprint:
                print(ordered_muon_points_Z)
        if YNprint:
            print(type(final_points[0]))

        return final_points

    def get_triggered_voxels_BETA_meh(self, voi: Volume, intersection_coordinates: List[Tensor]) -> List[Tensor]:
        """
        Calculates the triggered voxels based on the intersection points of the muon with the faces of the voxels.

        Args:
            voi (Volume): Volume of Interest containing voxel centers.
            intersection_coordinates (List[Tensor]): List of intersection coordinates for each muon.

        Returns:
            List[Tensor]: A list where each element is a tensor of shape (N, 3) containing the indices
                        of the triggered voxels for one muon.
        """
        triggered_voxels_list = []  # Lista que almacenará los tensores de índices

        for muon in intersection_coordinates:  # Iteramos sobre cada muón
            muon_voxels = []  # Lista para almacenar los índices de este muón

            for i, vox_idx in enumerate(muon):
                if i == 0:
                    cx0, cy0, cz0 = vox_idx  # Punto de entrada del muón
                    continue

                cx, cy, cz = vox_idx  # Punto de intersección actual

                # Calcular el punto medio entre dos intersecciones consecutivas
                X = torch.round((cx0 + cx) / 2, decimals=0)
                Y = torch.round((cy0 + cy) / 2, decimals=0)
                Z = torch.round((cz0 + cz) / 2, decimals=0)

                # Convertir a índice de voxel
                x_idx = int(np.floor((X + 450) / 30))
                y_idx = int(np.floor((Y + 300) / 30))
                z_idx = int(np.floor((Z + 1500) / 30))

                # Asegurar que los índices están dentro del rango permitido
                x_idx = max(0, min(x_idx, voi.voxel_centers.shape[0] - 1))
                y_idx = max(0, min(y_idx, voi.voxel_centers.shape[1] - 1))
                z_idx = max(0, min(z_idx, voi.voxel_centers.shape[2] - 1))

                muon_voxels.append([x_idx, y_idx, z_idx])  # Guardar los índices

                cx0, cy0, cz0 = cx, cy, cz  # Actualizar punto anterior

            if muon_voxels:  # Solo agregamos si hay voxeles activados
                triggered_voxels_list.append(torch.tensor(muon_voxels, dtype=torch.int32))
        if YNprint:
            print(type(triggered_voxels_list[0]))
        return triggered_voxels_list

    def get_triggered_voxels_BETA(self, voi: Volume, intersection_coordinates: List[Tensor]) -> List[Tensor]:
        """
        -> For one muon olnly (as it is going to be implemented on the for loop e¡in the intercection_coordinates function) <-
        Optimized calculation of triggered voxels using the intersection coordinates

        Args:
            voi (Volume): Volume of Interest containing voxel centers.
            intersection_coordinates (List[Tensor]): List of tensors,
                where each tensor has shape (mi, 3), representing intersection points for one muon.

        Returns:
            List[Tensor]: A list where each element is a tensor (Ni, 3) with the triggered voxel indices.
        """
        triggered_voxels_list = []

        for i, muon_intersection_coordinates in enumerate(intersection_coordinates):
            if muon_intersection_coordinates.shape[0] < 2:  # A muon with less than 2 intersections points doesn't trigger any voxels
                triggered_voxels_list.append(torch.empty((0, 3), dtype=torch.int32))
                continue

            # We calculate the point in between the two intersection points
            points_mid = (muon_intersection_coordinates[:-1] + muon_intersection_coordinates[1:]) / 2
            if i == 3:
                print(points_mid)

            # Now we transform the coordinates of that middle point into the indices of the triggered voxel
            x_idx = ((points_mid[:, 0] + abs(voi.xyz_min[0])) / voi.vox_width).floor().clamp(0, voi.voxel_centers.shape[0] - 1).to(torch.int32)
            y_idx = ((points_mid[:, 1] + abs(voi.xyz_min[1])) / voi.vox_width).floor().clamp(0, voi.voxel_centers.shape[1] - 1).to(torch.int32)
            z_idx = ((points_mid[:, 2] + abs(voi.xyz_min[2])) / voi.vox_width).floor().clamp(0, voi.voxel_centers.shape[2] - 1).to(torch.int32)
            # x_idx = ((points_mid[:, 0] + abs(voi.xyz_min[0])) / voi.vox_width).floor().to(torch.int32)
            # y_idx = ((points_mid[:, 1] + abs(voi.xyz_min[1])) / voi.vox_width).floor().to(torch.int32)
            # z_idx = ((points_mid[:, 2] + abs(voi.xyz_min[2])) / voi.vox_width).floor().to(torch.int32)

            indices = torch.stack([x_idx, y_idx, z_idx], dim=1)

            # We keep the tensor (Ni, 3) and append it to the list
            triggered_voxels_list.append(torch.unique(torch.stack([x_idx, y_idx, z_idx], dim=1), dim=0))

            # Update the M tensor (counts of voxel hits)
            # self.M_voxels.index_put_((x_idx, y_idx, z_idx), self.M_voxels[x_idx, y_idx, z_idx] + 1)
            # Example tensor of indices (N, 3) for 3D voxel indices
            # indices = torch.tensor([[0, 1, 2], [1, 2, 3], [0, 0, 1]]) # (N, 3)

            # Create a 3D volume of shape (X, Y, Z), initialized to zero
            # volume_shape = (5, 5, 5) # Example volume size
            # volume = torch.zeros(volume_shape, dtype=torch.int32)

            # Convert indices to a format suitable for scatter_add_
            flat_indices = indices.T  # Convert to list of indices per dimension

            # Increment volume at given indices
            self._M_voxels.index_put_(tuple(flat_indices), self._M_voxels[tuple(flat_indices)] + 1)
            # self.set_M_voxels().index_put_((x_idx, y_idx, z_idx), self.set_M_voxels()[x_idx, y_idx, z_idx] + 1)
            # self.set_M_voxels()[x_idx][y_idx][z_idx]+=1

        if YNprint:
            print(f"Intersection coordinates type: {type(intersection_coordinates)}, Triggered type: {type(triggered_voxels_list)}")
            print(
                f"Intersection coordinates element type: {type(intersection_coordinates[0])}, Triggered voxels element type: {type(triggered_voxels_list[0])}"
            )
            print(self._M_voxels)

        return triggered_voxels_list

    def get_L_T_length(self, intersection_coordinates: List[Tensor]) -> List[Tensor]:
        """
        -> For one muon olnly (as it is going to be implemented on the for loop e¡in the intercection_coordinates function) <-
        Optimized calculation ofL and T length using the intersection coordinates

        Args:
            voi (Volume): Volume of Interest containing voxel centers.
            intersection_coordinates (List[Tensor]): List of tensors,
                where each tensor has shape (mi, 3), representing intersection points for one muon.
            triggered_voxels (List[Tensor]): List of tensors, where each tensor has shape (mi, 3), representing indices of the triggered voxels for one muon.

        Returns:

        """
        # Puedo obtener una lista de tensores, cada tensor tiene dos elementos, L y T para cada muon, la longitud del tensor debe ser igual al numero de triggered voxels
        path_length_LT = []
        for i, muon_intersection_coordinates in enumerate(intersection_coordinates):
            if muon_intersection_coordinates.shape[0] < 2:  # A muon with less than 2 intersections points
                path_length_LT.append(torch.empty((0, 1), dtype=torch.int32))
                continue

            # Calcular L: distancia entre cada punto y el siguiente
            diffs_L = muon_intersection_coordinates[:-1] - muon_intersection_coordinates[1:]  # (N-1, 3)
            L = torch.norm(diffs_L, dim=1)  # (N-1)

            # Calcular T: distancia entre cada punto y el último
            last_point = muon_intersection_coordinates[-1].unsqueeze(0)  # (1, 3)
            diffs_T = muon_intersection_coordinates[:-1] - last_point  # (N-1, 3)
            T = torch.norm(diffs_T, dim=1)  # (N-1)

            # verifico que esté todo bien calculado:
            if sum(L) == T[0]:
                print("Verificado")
            else:
                print("Problema")
                print(sum(L), T[0])
                print(self.all_poca[i])

            # Unimos L y T en un tensor (N-1, 2)
            LT_tensor = torch.stack((L, T), dim=1)

            path_length_LT.append(LT_tensor)

        return path_length_LT

    # ---------------------------- #
    # New version of the functions #
    # ---------------------------- #

    def _compute_intersection_coordinates_BEFORE(
        self, voi: Volume, xyz_enters_voi: Tensor, xyz_exits_voi: Tensor, tracks_in: Tensor, tracks_out: Tensor, all_poca: Tensor
    ) -> Tensor:
        # Extract the volume limits to determine the planes' positions (de divisions of the volume into voxels)
        x_min, y_min, z_min = voi.xyz_min
        x_max, y_max, z_max = voi.xyz_max

        # Determine the number of voxels along each axis
        nx, ny, nz = voi.n_vox_xyz

        # Define the coordinate planes along the three axes
        planes_X = torch.tensor(np.arange(x_min, x_max + nx, voi.vox_width))  # (nx)
        planes_Y = torch.tensor(np.arange(y_min, y_max + ny, voi.vox_width))  # (ny)
        planes_Z = torch.tensor(np.arange(z_min, z_max + nz, voi.vox_width))  # (nz)

        # === Compute intersection points with planes for the entering track === #

        # Compute intersections with X = constant planes
        t_coord_X_in = (planes_X - xyz_enters_voi[0]) / tracks_in[0]
        y_coord_X_in = xyz_enters_voi[1] + t_coord_X_in * tracks_in[1]
        z_coord_X_in = xyz_enters_voi[2] + t_coord_X_in * tracks_in[2]
        intersection_points_X_in = torch.stack((planes_X, y_coord_X_in, z_coord_X_in), dim=-1)

        # Compute intersections with Y = constant planes
        t_coord_Y_in = (planes_Y - xyz_enters_voi[1]) / tracks_in[1]
        x_coord_Y_in = xyz_enters_voi[0] + t_coord_Y_in * tracks_in[0]
        z_coord_Y_in = xyz_enters_voi[2] + t_coord_Y_in * tracks_in[2]
        intersection_points_Y_in = torch.stack((x_coord_Y_in, planes_Y, z_coord_Y_in), dim=-1)

        # Compute intersections with Z = constant planes
        t_coord_Z_in = (planes_Z - xyz_enters_voi[2]) / tracks_in[2]
        x_coord_Z_in = xyz_enters_voi[0] + t_coord_Z_in * tracks_in[0]
        y_coord_Z_in = xyz_enters_voi[1] + t_coord_Z_in * tracks_in[1]
        intersection_points_Z_in = torch.stack((x_coord_Z_in, y_coord_Z_in, planes_Z), dim=-1)

        # === Compute intersection points for the exiting track === #
        t_coord_X_out = (planes_X - xyz_exits_voi[0]) / tracks_out[0]
        y_coord_X_out = xyz_exits_voi[1] + t_coord_X_out * tracks_out[1]
        z_coord_X_out = xyz_exits_voi[2] + t_coord_X_out * tracks_out[2]
        intersection_points_X_out = torch.stack((planes_X, y_coord_X_out, z_coord_X_out), dim=-1)

        t_coord_Y_out = (planes_Y - xyz_exits_voi[1]) / tracks_out[1]
        x_coord_Y_out = xyz_exits_voi[0] + t_coord_Y_out * tracks_out[0]
        z_coord_Y_out = xyz_exits_voi[2] + t_coord_Y_out * tracks_out[2]
        intersection_points_Y_out = torch.stack((x_coord_Y_out, planes_Y, z_coord_Y_out), dim=-1)

        t_coord_Z_out = (planes_Z - xyz_exits_voi[2]) / tracks_out[2]
        x_coord_Z_out = xyz_exits_voi[0] + t_coord_Z_out * tracks_out[0]
        y_coord_Z_out = xyz_exits_voi[1] + t_coord_Z_out * tracks_out[1]
        intersection_points_Z_out = torch.stack((x_coord_Z_out, y_coord_Z_out, planes_Z), dim=-1)

        # Combine all intersection points for each track
        total_intersection_points_in = torch.cat((intersection_points_X_in, intersection_points_Y_in, intersection_points_Z_in), dim=0)
        total_intersection_points_out = torch.cat((intersection_points_X_out, intersection_points_Y_out, intersection_points_Z_out), dim=0)

        # Compute masks based on enter, exit and POCA position
        z_inicial = xyz_enters_voi[2]
        z_final = xyz_exits_voi[2]
        z_POCA = all_poca[2]

        mask_in = (total_intersection_points_in[:, 2] <= z_inicial) & (total_intersection_points_in[:, 2] >= z_POCA)
        mask_out = (total_intersection_points_out[:, 2] <= z_POCA) & (total_intersection_points_out[:, 2] >= z_final)

        filtered_points_in = total_intersection_points_in[mask_in]
        filtered_points_out = total_intersection_points_out[mask_out]

        # Concatenate and order by z coordinate
        muon_points = torch.cat((filtered_points_in, filtered_points_out), dim=0)
        ordered_muon_points = muon_points[torch.argsort(muon_points[:, 2], descending=True)]
        # print(f"Tamaño numero de puntos interseccion: {ordered_muon_points.size()[0]}")

        # Some intersections might be very close to each other (for expample if the intersection point is right at the corner of the voxel)
        # Define a small tolerance to consider points as duplicates
        tolerance = 1  # Adjust based on required precision

        # Compute the difference between consecutive points
        diffs = torch.norm(ordered_muon_points[1:] - ordered_muon_points[:-1], dim=1)
        # print(f"Diferencias: {diffs}")

        # Keep the first point and those whose difference with the previous one is greater than the tolerance
        mask = torch.cat((torch.tensor([True], device=diffs.device), diffs > tolerance))

        # Filter out redundant points
        filtered_muon_points = ordered_muon_points[mask]
        # print(f"\tTamaño numero de puntos interseccion: {filtered_muon_points.size()[0]}")

        return filtered_muon_points
        # return ordered_muon_points

    def _compute_intersection_coordinates(self, voi: Volume, xyz_point: Tensor, xyz_poca: Tensor, incoming: bool = True) -> Tensor:
        # We need the direction
        if incoming:
            track = xyz_point - xyz_poca
            z_inicial = xyz_point[2]  # Mask (see below)
            z_final = xyz_poca[2]  # Mask (see below)
        else:
            track = xyz_poca - xyz_point
            z_inicial = xyz_poca[2]  # Mask (see below)
            z_final = xyz_point[2]  # Mask (see below)

        # Extract the volume limits to determine the planes' positions (de divisions of the volume into voxels)
        x_min, y_min, z_min = voi.xyz_min
        x_max, y_max, z_max = voi.xyz_max

        # Determine the number of voxels along each axis
        nx, ny, nz = voi.n_vox_xyz

        # Define the coordinate planes along the three axes
        self.planes_X = torch.tensor(np.arange(x_min, x_max + nx, voi.vox_width))  # (nx)
        self.planes_Y = torch.tensor(np.arange(y_min, y_max + ny, voi.vox_width))  # (ny)
        self.planes_Z = torch.tensor(np.arange(z_min, z_max + nz, voi.vox_width))  # (nz)

        # === Compute intersection points with planes for the entering track === #

        # Compute intersections with X = constant planes
        t_coord_X_in = (self.planes_X - xyz_point[0]) / track[0]
        y_coord_X_in = xyz_point[1] + t_coord_X_in * track[1]
        z_coord_X_in = xyz_point[2] + t_coord_X_in * track[2]
        intersection_points_X_in = torch.stack((self.planes_X, y_coord_X_in, z_coord_X_in), dim=-1)

        # Compute intersections with Y = constant planes
        t_coord_Y_in = (self.planes_Y - xyz_point[1]) / track[1]
        x_coord_Y_in = xyz_point[0] + t_coord_Y_in * track[0]
        z_coord_Y_in = xyz_point[2] + t_coord_Y_in * track[2]
        intersection_points_Y_in = torch.stack((x_coord_Y_in, self.planes_Y, z_coord_Y_in), dim=-1)

        # Compute intersections with Z = constant planes
        t_coord_Z_in = (self.planes_Z - xyz_point[2]) / track[2]
        x_coord_Z_in = xyz_point[0] + t_coord_Z_in * track[0]
        y_coord_Z_in = xyz_point[1] + t_coord_Z_in * track[1]
        intersection_points_Z_in = torch.stack((x_coord_Z_in, y_coord_Z_in, self.planes_Z), dim=-1)

        # Combine all intersection points for each track
        total_intersection_points = torch.cat((intersection_points_X_in, intersection_points_Y_in, intersection_points_Z_in), dim=0)

        # Compute masks based on enter, exit and POCA position
        mask = (total_intersection_points[:, 2] <= z_inicial) & (total_intersection_points[:, 2] >= z_final)

        filtered_points = total_intersection_points[mask]

        # Concatenate and order by z coordinate
        ordered_muon_points = filtered_points[torch.argsort(filtered_points[:, 2], descending=True)]
        # print(f"Tamaño numero de puntos interseccion: {ordered_muon_points.size()[0]}")

        # Some intersections might be very close to each other (for expample if the intersection point is right at the corner of the voxel)
        # Define a small tolerance to consider points as duplicates
        tolerance = 0.01 * voi.vox_width

        # Compute the difference between consecutive points
        diffs = torch.norm(ordered_muon_points[1:] - ordered_muon_points[:-1], dim=1)
        # print(f"Diferencias: {diffs}")

        # Keep the first point and those whose difference with the previous one is greater than the tolerance
        mask = torch.cat((torch.tensor([True], device=diffs.device), diffs > tolerance))

        # Filter out redundant points
        filtered_muon_points = ordered_muon_points[mask]
        # print(f"\tTamaño numero de puntos interseccion: {filtered_muon_points.size()[0]}")

        return filtered_muon_points

    def _compute_triggered_voxels(self, muon_intersection_coordinates: Tensor, voi: Volume) -> Tensor:
        if muon_intersection_coordinates.shape[0] < 2:  # A muon with less than 2 intersections points doesn't trigger any voxels
            triggered_voxels_list = torch.empty((0, 3), dtype=torch.int32)
        else:
            # We calculate the point in between the two intersection points
            points_mid = (muon_intersection_coordinates[:-1] + muon_intersection_coordinates[1:]) / 2

            # Now we transform the coordinates of that middle point into the indices of the triggered voxel
            x_idx = ((points_mid[:, 0] + abs(voi.xyz_min[0])) / voi.vox_width).floor().clamp(0, voi.voxel_centers.shape[0] - 1).to(torch.int32)
            y_idx = ((points_mid[:, 1] + abs(voi.xyz_min[1])) / voi.vox_width).floor().clamp(0, voi.voxel_centers.shape[1] - 1).to(torch.int32)
            z_idx = ((points_mid[:, 2] + abs(voi.xyz_min[2])) / voi.vox_width).floor().clamp(0, voi.voxel_centers.shape[2] - 1).to(torch.int32)
            # Si pongo .round() empeoro la situacion porque caluclo menos voxels aun
            # x_idx = ((points_mid[:, 0] + abs(voi.xyz_min[0])) / voi.vox_width).round().clamp(0, voi.voxel_centers.shape[0] - 1).to(torch.int32)
            # y_idx = ((points_mid[:, 1] + abs(voi.xyz_min[1])) / voi.vox_width).round().clamp(0, voi.voxel_centers.shape[1] - 1).to(torch.int32)
            # z_idx = ((points_mid[:, 2] + abs(voi.xyz_min[2])) / voi.vox_width).round().clamp(0, voi.voxel_centers.shape[2] - 1).to(torch.int32)

            indices = torch.stack([x_idx, y_idx, z_idx], dim=1)

            # We keep the tensor (Ni, 3) and append it to the list
            triggered_voxels_list = torch.unique(torch.stack([x_idx, y_idx, z_idx], dim=1), dim=0)
            # triggered_voxels_list = torch.stack([x_idx, y_idx, z_idx], dim=1)

            # Convert indices to a format suitable for scatter_add_
            flat_indices = indices.T  # Convert to list of indices per dimension

            # Increment volume at given indices
            self._M_voxels.index_put_(tuple(flat_indices), self._M_voxels[tuple(flat_indices)] + 1)

        return triggered_voxels_list

    def _compute_L_T_length(self, muon_intersection_coordinates: Tensor, idx: int) -> Tensor:
        # I use idx to print the poca point of the muon, just to see if there is a problem or not
        if muon_intersection_coordinates.shape[0] < 2:  # A muon with less than 2 intersections points
            path_length_LT = torch.empty((0, 1), dtype=torch.int32)
        else:
            # Calcular L: distancia entre cada punto y el siguiente
            diffs_L = muon_intersection_coordinates[:-1] - muon_intersection_coordinates[1:]  # (N-1, 3)
            L = torch.norm(diffs_L, dim=1)  # (N-1)

            # Calcular T: distancia entre cada punto y el último
            last_point = muon_intersection_coordinates[-1].unsqueeze(0)  # (1, 3)
            diffs_T = muon_intersection_coordinates[:-1] - last_point  # (N-1, 3)
            T = torch.norm(diffs_T, dim=1)  # (N-1)

            # Unimos L y T en un tensor (N-1, 2)
            LT_tensor = torch.stack((L, T), dim=1)

            path_length_LT = LT_tensor

            if YNprint:
                if (
                    sum(L).round() == T[0].round()
                ):  # this means the sum of every L-length equals to the first T-length calculated (the 1º one corresponds to the entire trace within the volumen)
                    print(f"--> GOOD muon {idx}")
                    # print(sum(L), T[0])
                    # print(self.all_poca[idx])

                else:
                    print(f"* muon {idx} NOT right")
                    print(sum(L), T[0])
                    print(self.all_poca[idx])

        return path_length_LT

    def _compute_weight_matrix(self, voxels: Tensor, LT: Tensor, idx: int) -> None:
        """
        Calcula la matriz de pesos W para el algoritmo de Expectación-Maximización.
        """

        if len(voxels) > 0 and len(LT) > 0:
            L = LT[:, 0].float()  # Convertir a float32
            T = LT[:, 1].float()  # Convertir a float32

            # Índices de los voxeles en el grid
            x_idx, y_idx, z_idx = voxels[:, 0], voxels[:, 1], voxels[:, 2]

            # Calcular los elementos de W
            self._W[idx, x_idx, y_idx, z_idx, 0, 0] = L
            self._W[idx, x_idx, y_idx, z_idx, 0, 1] = (L**2) / 2 + L * T
            self._W[idx, x_idx, y_idx, z_idx, 1, 0] = (L**2) / 2 + L * T
            self._W[idx, x_idx, y_idx, z_idx, 1, 1] = (L**3) / 3 + (L**2) * T + L * (T**2)

    def process_all_muons(self) -> None:
        """
        Performs the full processing of muons in a single loop:
        1. Computes intersections with the voxels
        2. Determines the activated voxels
        3. Computes the path lengths L and T
        4. Computes the W matrix

        Stores the results in the corresponding attributes.
        """
        intersection_coordinates_list = []
        triggered_voxels_BETA_list = []
        path_length_LT_list = []

        Ni, Nj, Nk = self.voi.n_vox_xyz  # Voxel grid dimensions

        # Identify problematic muons (with POCA at [0,0,0])
        problematic_muons_mask = (self._all_poca == torch.tensor([0.0, 0.0, 0.0], device=self._all_poca.device)).all(dim=1)

        # Filter out problematic POCA points
        self._valid_poca = self._all_poca[~problematic_muons_mask]  # Keeps only valid POCA points
        self._valid_xyz_enters_voi = self._xyz_enters_voi[~problematic_muons_mask]
        self._valid_xyz_exits_voi = self._xyz_exits_voi[~problematic_muons_mask]
        self._valid_tracks_in = self.tracking.tracks_in[~problematic_muons_mask]
        self._valid_tracks_out = self.tracking.tracks_out[~problematic_muons_mask]

        # This will be used for 3D visualization
        # self._last_entry_point = torch.zeros(size=self._valid_xyz_enters_voi.size(), device=self._valid_poca.device)
        # self._first_exit_point = torch.zeros(size=self._valid_xyz_exits_voi.size(), device=self._valid_poca.device)

        n_events = len(self._valid_poca)  # Number of valid muons

        # Initialize W (weight matrix) with zeros
        self._W = torch.zeros(n_events, Ni, Nj, Nk, 2, 2, device=self._valid_poca.device)

        # Iterate over all valid muons
        for i in range(len(self._valid_poca)):
            # Step 1: Compute intersections for the incoming segment
            intersec_coords_in = self._compute_intersection_coordinates(voi=self.voi, xyz_point=self._valid_xyz_enters_voi[i], xyz_poca=self._valid_poca[i])
            # self._last_entry_point[i] = intersec_coords_in[-1]

            # Compute intersections for the outgoing segment
            intersec_coords_out = self._compute_intersection_coordinates(
                voi=self.voi, xyz_point=self._valid_xyz_exits_voi[i], xyz_poca=self._valid_poca[i], incoming=False
            )
            # self._first_exit_point[i] = intersec_coords_out[0]

            # Check if both the last incoming and first outgoing points lie on lateral (X or Y) planes
            last_point = intersec_coords_in[-1]
            first_point = intersec_coords_out[0]

            if (last_point[0] in self.planes_X or last_point[1] in self.planes_Y) and (first_point[0] in self.planes_X or first_point[1] in self.planes_Y):
                # If both are on lateral faces, remove the last point from the incoming set
                intersec_coords_in = intersec_coords_in[:-1]
                # And remove the first point from the outgoing set
                intersec_coords_out = intersec_coords_out[1:]

            # Concatenate both segments
            intersection_coordinates = torch.cat((intersec_coords_in, intersec_coords_out), dim=0)
            intersection_coordinates_list.append(intersection_coordinates)

            # Step 2: Compute which voxels were triggered (intersected)
            triggered_voxels = self._compute_triggered_voxels(intersection_coordinates, self.voi)  # No repeated voxels
            triggered_voxels_BETA_list.append(triggered_voxels)

            # Step 3: Compute the path lengths L and T through the voxels
            L_T_values = self._compute_L_T_length(intersection_coordinates, i)
            path_length_LT_list.append(L_T_values)

            # Sanity check: number of path length entries should match number of triggered voxels
            if L_T_values.size()[0] == triggered_voxels.size()[0]:
                continue
            else:
                print(f"Revisar: muon {i}")  # Debug message if there's a mismatch

            # Step 4: Compute the weight matrix W for this muon
            self._compute_weight_matrix(triggered_voxels, L_T_values, i)

        # Store the computed results in instance attributes
        self._intersection_coordinates = intersection_coordinates_list
        self._triggered_voxels_BETA = triggered_voxels_BETA_list
        self._path_length_LT = path_length_LT_list

    # ---------------------------- #
    #                              #
    # ---------------------------- #

    def draw_cube(self, ax: Axes3D, center: Tensor, side: float, color: str = "blue", alpha: float = 0.1) -> None:
        """Dibuja un cubo 3D centrado en `center` con longitud de lado `side`"""
        x, y, z = center
        s = side / 2  # Mitad del lado

        # Definir los 8 vértices del cubo
        vertices = np.array(
            [
                [x - s, y - s, z - s],
                [x + s, y - s, z - s],
                [x + s, y + s, z - s],
                [x - s, y + s, z - s],
                [x - s, y - s, z + s],
                [x + s, y - s, z + s],
                [x + s, y + s, z + s],
                [x - s, y + s, z + s],
            ]
        )

        # Definir las 6 caras del cubo
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],  # Inferior
            [vertices[j] for j in [4, 5, 6, 7]],  # Superior
            [vertices[j] for j in [0, 1, 5, 4]],  # Frontal
            [vertices[j] for j in [2, 3, 7, 6]],  # Trasera
            [vertices[j] for j in [0, 3, 7, 4]],  # Izquierda
            [vertices[j] for j in [1, 2, 6, 5]],  # Derecha
        ]

        # Crear colección de polígonos para el cubo
        cube = Poly3DCollection(faces, alpha=alpha, edgecolor="k")
        cube.set_facecolor(color)
        ax.add_collection3d(cube)  # type: ignore

    def plot_3D_inters_all_muons(self, idx_muons: np.ndarray) -> None:
        """
        Plots the trajectories of multiple muons along with their intersection points in 3D.

        Args:
            num_muons (int): Number of muons to plot.
        """
        # Randomly select `num_muons` events from available data
        # num_events = np.random.choice(self.xyz_enters_voi.size(0), num_muons, replace=False)
        print(idx_muons)

        # Create an interactive 3D figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Define distinct colors for each muon trajectory
        colores = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])

        x_min, y_min, z_min = self.voi.xyz_min
        x_max, y_max, z_max = self.voi.xyz_max

        # Iterate over selected muon events
        for i, event in enumerate(idx_muons):
            intersection_points = self._intersection_coordinates[event]  # Tensor con puntos de intersección (3)

            # Retrieve key points for each muon
            entry_point = self._valid_xyz_enters_voi[event]
            exit_point = self._valid_xyz_exits_voi[event]
            poca_point = self._valid_poca[event]

            # Assign a unique color for each muon track
            muons_colors = colores[i % len(colores)]

            # === Plot the incoming and outgoing trajectory === #
            # # Incoming track (entry to POCA)
            ax.plot(
                [entry_point[0], poca_point[0]],
                [entry_point[1], poca_point[1]],
                [entry_point[2], poca_point[2]],
                color=muons_colors,
                linestyle="-",
                label=f"Muon {i} - Incoming",
                alpha=0.6,
            )

            # Outgoing track (POCA to exit)
            ax.plot(
                [poca_point[0], exit_point[0]],
                [poca_point[1], exit_point[1]],
                [poca_point[2], exit_point[2]],
                color=muons_colors,
                linestyle="--",
                label=f"Muon {i} - Outgoing",
                alpha=0.6,
            )

            # === Plot intersection points (where muon interacts with voxel faces) === #
            # ax.scatter(intersection_points[i][:, 0], intersection_points[i][:, 1], intersection_points[i][:, 2], color=color_traza, marker=".", label=f"Muon {i} - Intersections")
            ax.scatter(
                intersection_points[:, 0],
                intersection_points[:, 1],
                intersection_points[:, 2],
                color=muons_colors,
                marker=".",
                edgecolors="black",
                s=45,
                label=f"Muon {i} - Intersections",
            )  # type: ignore[misc]

            # === Mark key points (entry, exit, and POCA) === #
            # ax.scatter(*entry_point, color=color_traza, marker="v", edgecolors='black')
            # ax.scatter(*exit_point, color=color_traza, marker="^", edgecolors='black')
            # # ax.scatter(*poca_point, color="black", marker="o", label=f"Muon {i} - POCA")
            ax.scatter(
                entry_point[0], entry_point[1], entry_point[2], color=muons_colors, marker="v", edgecolors="black", s=45, label=f"Muon {i} - entry point"
            )  # type: ignore[misc]
            ax.scatter(exit_point[0], exit_point[1], exit_point[2], color=muons_colors, marker="^", edgecolors="black", s=45, label=f"Muon {i} - exit point")  # type: ignore[misc]
            ax.scatter(poca_point[0], poca_point[1], poca_point[2], color="black", marker="o", s=50, label=f"Muon {i} - POCA")  # type: ignore[misc]

            # # === Draw triggered voxels === #
            # if self.triggered_voxels[event].shape[0] > 0:
            #     for i, vox_idx in enumerate(self.triggered_voxels[event]):
            #         ix, iy, iz = vox_idx  # Indices in the Volume of Interest (VOI)
            #         voxel_center = self.voi.voxel_centers[ix, iy, iz]
            #         self.draw_cube(ax, center=voxel_center, side=self.voi.vox_width, color='red', alpha=0.2)
            if self.triggered_voxels_BETA[event].shape[0] > 0:
                for i, vox_idx in enumerate(self.triggered_voxels_BETA[event]):
                    ix, iy, iz = vox_idx  # Indices in the Volume of Interest (VOI)
                    voxel_center = self.voi.voxel_centers[ix, iy, iz]
                    self.draw_cube(ax, center=voxel_center, side=self.voi.vox_width, color=muons_colors, alpha=0.2)

        # Set axis labels
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")
        ax.set_zlabel("Z [mm]")  # type: ignore[attr-defined]
        ax.set_title(f"{idx_muons} muons")

        # Adjust plot limits to cover the full volume
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)  # type: ignore[attr-defined]

        # Display legend
        ax.legend()

        # Show the 3D plot
        plt.show()

    @staticmethod
    def compute_path_length_in_out(
        poca_points: Tensor,
        xyz_enters_voi: Tensor,
        xyz_exits_voi: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        path_length_in = torch.sqrt(torch.sum((xyz_enters_voi - poca_points) ** 2, dim=1))

        path_length_out = torch.sqrt(torch.sum((xyz_exits_voi - poca_points) ** 2, dim=1))
        return path_length_in, path_length_out

    @staticmethod
    def compute_straight_path_length(
        xyz_enters_voi: Tensor,
        xyz_exits_voi: Tensor,
    ) -> Tensor:
        """Computes the muon path length between the entry and
        exit points in the VOI, assuming a straight path.

        Args:
            xyz_enters_voi (Tensor): The muon entry point in the VOI.
            xyz_exits_voi (Tensor): The muon exit point in the VOI.

        Returns:
            Tensor: The path length between the entry and exit points.
        """

        return torch.sqrt(torch.sum((xyz_enters_voi - xyz_exits_voi) ** 2, dim=1))

    @staticmethod
    def recompute_point(
        xyz_in_voi: Tensor,
        voi: Volume,
        theta_xy: Tensor,
        pm: int = -1,
    ) -> Tensor:
        """
        Recomputes the muon entry point within a volume of interest (VOI),
        correcting its coordinates when it enters from the left/right side of the VOI in x and/or y.

        The function accounts for boundary conditions along the x and y axes and adjusts the z-coordinate
        accordingly based on angular projections specified by `theta_xy_in`.

        Args:
            xyz_in_voi (Tensor): A tensor of shape `(N, 3)` representing the (x, y, z) coordinates
                of N muons when reaching the top/bottom of the VOI.
            voi (Volume): The Volume of interest.
            theta_xy_in (Tensor): A tensor of shape `(2,)` containing the angular projections of the
                muon in radians:
                    - `theta_xy_in[0]`: Angle in the x-z plane.
                    - `theta_xy_in[1]`: Angle in the y-z plane.

        Returns:
            Tensor: A tensor of shape `(N, 3)` with updated coordinates for each point,
            adjusted to lie within the bounds of the VOI.
        """

        xyz_in_voi_new = xyz_in_voi.clone()

        # Entry conditions in x and y dimensions
        enters_left_x = xyz_in_voi[:, 0] < voi.xyz_min[0]
        enters_right_x = xyz_in_voi[:, 0] > voi.xyz_max[0]
        enters_in_x = ~(enters_left_x | enters_right_x)

        enters_left_y = xyz_in_voi[:, 1] < voi.xyz_min[1]
        enters_right_y = xyz_in_voi[:, 1] > voi.xyz_max[1]
        enters_in_y = ~(enters_left_y | enters_right_y)

        # Calculate deltas
        delta_x = torch.where(
            ~enters_in_x,
            torch.min(torch.abs(xyz_in_voi[:, 0] - voi.xyz_min[0]), torch.abs(xyz_in_voi[:, 0] - voi.xyz_max[0])),
            0.0,
        )
        delta_y = torch.where(
            ~enters_in_y,
            torch.min(torch.abs(xyz_in_voi[:, 1] - voi.xyz_min[1]), torch.abs(xyz_in_voi[:, 1] - voi.xyz_max[1])),
            0.0,
        )

        # Helper function for coordinate corrections
        def correct_coords(mask: Tensor, coord_idx: int, target_value: Optional[float] = None, adjust_fn: Optional[Callable] = None) -> None:
            xyz_in_voi_new[:, coord_idx] = torch.where(
                mask, adjust_fn(xyz_in_voi_new[:, coord_idx]) if adjust_fn else target_value, xyz_in_voi_new[:, coord_idx]
            )

        # Correct coordinates based on entry conditions
        correct_coords(enters_left_x, 0, voi.xyz_min[0].cpu().item())
        correct_coords(enters_right_x, 0, voi.xyz_max[0].cpu().item())
        correct_coords(enters_left_y, 1, voi.xyz_min[1].cpu().item())
        correct_coords(enters_right_y, 1, voi.xyz_max[1].cpu().item())

        # z and y corrections for x boundary crossings
        x_cross_mask = enters_left_x | enters_right_x
        correct_coords(x_cross_mask, 2, None, lambda z: z + (pm * torch.abs(delta_x / torch.tan(theta_xy[0]))))
        correct_coords(x_cross_mask & enters_in_y, 1, None, lambda y: y + pm * (torch.abs(delta_x / torch.tan(theta_xy[0])) * torch.tan(theta_xy[1])))

        # z and x corrections for y boundary crossings
        y_cross_mask = enters_left_y | enters_right_y
        correct_coords(y_cross_mask & enters_in_x, 2, None, lambda z: z + pm * torch.abs(delta_y / torch.tan(theta_xy[1])))
        correct_coords(y_cross_mask & enters_in_x, 0, None, lambda x: x + pm * (torch.abs(delta_y / torch.tan(theta_xy[1])) * torch.tan(theta_xy[0])))

        return xyz_in_voi_new

    def plot_event(
        self,
        event: int,
        proj: str = "XZ",
        figname: Optional[str] = None,
        points: Optional[Tensor] = None,
    ) -> None:
        configure_plot_theme(font=font)  # type: ignore

        # Inidices and labels
        dim_map: Dict[str, Dict[str, Union[str, int]]] = {
            "XZ": {"x": 0, "y": 2, "xlabel": r"$x$ [mm]", "ylabel": r"$z$ [mm]", "dim": 1},
            "YZ": {"x": 1, "y": 2, "xlabel": r"$y$ [mm]", "ylabel": r"$z$ [mm]", "dim": 0},
        }

        # Data numpy
        points_in_np = self._valid_xyz_enters_voi.detach().cpu().numpy()
        points_out_np = self._valid_xyz_exits_voi.detach().cpu().numpy()
        track_in_np = self._valid_tracks_in.detach().cpu().numpy()[event]
        track_out_np = self._valid_tracks_out.detach().cpu().numpy()[event]

        # Y span
        y_span = abs(points_in_np[event, 2] - points_out_np[event, 2])

        fig, ax = plt.subplots(figsize=tracking_figsize)

        # Plot POCA point
        if self.poca is not None:
            if self._valid_poca[event, dim_map[proj]["x"]] != 0:  # type: ignore
                ax.scatter(
                    x=self._valid_poca[event, dim_map[proj]["x"]],  # type: ignore
                    y=self._valid_poca[event, dim_map[proj]["y"]],  # type: ignore
                    color="black",
                    label="POCA point",
                )

            # ax.scatter(
            #     x=self.xyz_in_out_voi[0][event, 0, dim_map[proj]["x"]],
            #     # x=self.xyz_in_out_voi_new[event, dim_map[proj]["x"]],
            #     y=self.xyz_in_out_voi[0][event, 0, dim_map[proj]["y"]],
            #     # y=self.xyz_in_out_voi_new[event, dim_map[proj]["y"]],
            #     color="red",
            #     label=r"Track$_{in}$ exit point",
            #     marker="x",
            # )

            ax.scatter(
                x=self._valid_xyz_enters_voi[event, dim_map[proj]["x"]],  # type: ignore
                y=self._valid_xyz_enters_voi[event, dim_map[proj]["y"]],  # type: ignore
                color="red",
                label=r"Track$_{in}$ entry point",
                marker="x",
                s=50,
            )

            # ax.scatter(
            # x=self.xyz_in_out_voi[1][event, 1, dim_map[proj]["x"]],
            # y=self.xyz_in_out_voi[1][event, 1, dim_map[proj]["y"]],
            # x=self.xyz_exits_voi[event, dim_map[proj]["x"]],
            # y=self.xyz_exits_voi[event, dim_map[proj]["x"]],
            # color="green",
            # label=r"Track$_{out}$ exit point",
            # marker="x",
            # )

            # ax.scatter(
            #     x=self.xyz_in_out_voi[1][event, 0, dim_map[proj]["x"]],
            #     y=self.xyz_in_out_voi[1][event, 0, dim_map[proj]["y"]],
            #     color="green",
            #     label=r"Track$_{out}$ entry point",
            #     marker="+",
            #     s=100,
            # )

            ax.scatter(
                x=self._valid_xyz_exits_voi[event, dim_map[proj]["x"]],  # type: ignore
                y=self._valid_xyz_exits_voi[event, dim_map[proj]["y"]],  # type: ignore
                color="green",
                label=r"Track$_{out}$ entry point",
                marker="x",
                s=50,
            )

        if points is not None:
            ax.scatter(
                x=points[:, dim_map[proj]["x"]],  # type: ignore
                y=points[:, dim_map[proj]["y"]],  # type: ignore
                color="black",
                label="points",
            )

        # if self.triggered_voxels[event].shape[0] > 0:
        #     n_trig_vox = f"# triggered voxels = {self.triggered_voxels[event].shape[0]}"
        if self.triggered_voxels_BETA[event].shape[0] > 0:
            n_trig_vox = f"# triggered voxels = {self.triggered_voxels_BETA[event].shape[0]}"
        else:
            n_trig_vox = "no voxels triggered"
        fig.suptitle(
            f"Tracking of event {event:,d}" + "\n" + r"$\delta\theta$ = " + f"{self.tracking.dtheta[event] * 180 / math.pi:.2f} deg, " + n_trig_vox,
            fontweight="bold",
            y=1.05,
        )

        # Plot voxel grid
        self.plot_voxel_grid(
            dim=dim_map[proj]["dim"],  # type: ignore
            # dim=1,  #type: ignore
            voi=self.voi,
            ax=ax,
        )

        # Plot triggered voxels
        # if self.triggered_voxels[event].shape[0] > 0:
        #     for i, vox_idx in enumerate(self.triggered_voxels[event]):
        #         ix, iy = vox_idx[dim_map[proj]["x"]], vox_idx[2]
        #         vox_x = self.voi.voxel_centers[ix, 0, 0, 0] if proj == "XZ" else self.voi.voxel_centers[0, ix, 0, 1]
        #         label = "Triggered voxel" if i == 0 else None
        #         ax.scatter(
        #             x=vox_x,
        #             y=self.voi.voxel_centers[0, 0, iy, 2],
        #             color="blue",
        #             label=label,
        #             alpha=0.3,
        #         )
        if self.triggered_voxels_BETA[event].shape[0] > 0:
            for i, vox_idx in enumerate(self.triggered_voxels_BETA[event]):
                ix, iy = vox_idx[dim_map[proj]["x"]], vox_idx[2]
                vox_x = self.voi.voxel_centers[ix, 0, 0, 0] if proj == "XZ" else self.voi.voxel_centers[0, ix, 0, 1]
                label = "Triggered voxel" if i == 0 else None
                ax.scatter(
                    x=vox_x,
                    y=self.voi.voxel_centers[0, 0, iy, 2],
                    color="blue",
                    label=label,
                    alpha=0.3,
                )

        # Plot tracks
        for point, track, label, pm, color in zip(
            (points_in_np[event], points_out_np[event]), (track_in_np, track_out_np), ("in", "out"), (1, -1), ("red", "green")
        ):
            ax.plot(
                [point[dim_map[proj]["x"]], point[dim_map[proj]["x"]] + track[dim_map[proj]["x"]] * y_span * pm],  # type: ignore
                [point[dim_map[proj]["y"]], point[dim_map[proj]["y"]] + track[dim_map[proj]["y"]] * y_span * pm],  # type: ignore
                alpha=0.6,
                color=color,
                linestyle="--",
                label=f"Fitted track {label}",
            )

        # Legend
        ax.legend(
            bbox_to_anchor=(1.0, 0.7),
        )

        # Save figure
        if figname is not None:
            plt.savefig(figname, bbox_inches="tight")
        plt.show()

    @staticmethod
    def get_all_poca(poca: POCA, tracking: TrackingMST) -> Tensor:
        all_poca = torch.zeros_like(tracking.tracks_in, device=tracking.tracks_in.device)
        all_poca[poca.full_mask] = poca.poca_points

        return all_poca

    @property
    def xyz_in_out_voi(self) -> Tuple[Tensor, Tensor]:
        """
        Coordinates of the muon incoming/outgoing track when reaching the top / bottom of the voi.
        Example:
            - Coordinate of the muon incoming track when reaching the top of the voi:
            `xyz_in_out_voi[0][:, 1]`
            - Coordinate of the muon incoming track when reaching the bottom of the voi:
            `xyz_in_out_voi[0][:, 0]`
            - Coordinate of the muon outgoing track when reaching the top of the voi:
            `xyz_in_out_voi[1][:, 1]`
            - Coordinate of the muon outgoing track when reaching the bottom of the voi:
            `xyz_in_out_voi[1][:, 0]`

        """
        if self._xyz_in_out_voi is None:
            self._xyz_in_out_voi = ASR._compute_xyz_in_out(
                points_in=self.tracking.points_in,
                points_out=self.tracking.points_out,
                voi=self.voi,
                theta_xy_in=(self.tracking.theta_xy_in[0], self.tracking.theta_xy_in[1]),
                theta_xy_out=(self.tracking.theta_xy_out[0], self.tracking.theta_xy_out[1]),
            )
        return self._xyz_in_out_voi

    @property
    def path_length_in_out(self) -> Tuple[Tensor, Tensor]:
        """Path length between the incoming tracks entry point in the voi and the poca point,
        and ath length between the outgoing tracks exit point in the voi and the poca point"""
        if self._path_length_in_out is None:
            self._path_length_in_out = self.compute_path_length_in_out(
                poca_points=self.all_poca,
                xyz_enters_voi=self.xyz_enters_voi,
                xyz_exits_voi=self.xyz_exits_voi,
            )
        return self._path_length_in_out

    # @property
    # def triggered_voxels(self) -> List[np.ndarray]:
    #     """Voxels triggered by the muon track"""
    #     if self._triggered_voxels is None:
    #         self._triggered_voxels = TrackingEM.get_triggered_voxels(
    #             points_in=self.tracking.points_in,
    #             points_out=self.tracking.points_out,
    #             voi=self.voi,
    #             theta_xy_in=(self.tracking.theta_xy_in[0], self.tracking.theta_xy_in[1]),
    #             theta_xy_out=(self.tracking.theta_xy_out[0], self.tracking.theta_xy_out[1]),
    #         )
    #     return self._triggered_voxels

    # @property
    # def triggered_voxels_BETA_old(self) -> List[Tensor]:
    #     """Voxels triggered by the muon track"""
    #     if self._triggered_voxels_BETA is None:
    #         if self._intersection_coordinates is None:
    #             self._intersection_coordinates = self.compute_intersection_coordinates_all_muons(
    #                 voi=self.voi,
    #                 xyz_enters_voi=self._xyz_enters_voi,
    #                 xyz_exits_voi=self._xyz_exits_voi,
    #                 tracks_in=self.tracking.tracks_in,
    #                 tracks_out=self.tracking.tracks_out,
    #                 all_poca=self._all_poca,
    #             )
    #         self._triggered_voxels_BETA = TrackingEM.get_triggered_voxels_BETA(self, voi=self.voi, intersection_coordinates=self._intersection_coordinates)
    #     return self._triggered_voxels_BETA

    @property
    def xyz_enters_voi(self) -> Tensor:
        """Coordinates of the muon incoming track when entering the voi"""
        if self._xyz_enters_voi is None:
            self._xyz_enters_voi = self.recompute_point(xyz_in_voi=self.xyz_in_out_voi[0][:, 1], voi=self.voi, theta_xy=self.tracking.theta_xy_in, pm=-1)
        return self._xyz_enters_voi

    @property
    def xyz_exits_voi(self) -> Tensor:
        """Coordinates of the muon outgoing track when exiting the voi"""
        if self._xyz_exits_voi is None:
            self._xyz_exits_voi = self.recompute_point(xyz_in_voi=self.xyz_in_out_voi[1][:, 0], voi=self.voi, theta_xy=self.tracking.theta_xy_out, pm=1)
        return self._xyz_exits_voi

    @property
    def all_poca(self) -> Tensor:
        """The poca points associated to ALL the tracks in self.tracking"""
        if self._all_poca is None:
            self._all_poca = self.get_all_poca(poca=self.poca, tracking=self.tracking)
        return self._all_poca

    # @property
    # def intersection_coordinates_old(self) -> List[Tensor]:
    #     """The intersection points of the track (incoming and outgoing) with the triggered voxels of all muons"""
    #     if self._intersection_coordinates is None:
    #         self._intersection_coordinates = self.compute_intersection_coordinates_all_muons(
    #             voi=self.voi,
    #             xyz_enters_voi=self._xyz_enters_voi,
    #             xyz_exits_voi=self._xyz_exits_voi,
    #             tracks_in=self.tracking.tracks_in,
    #             tracks_out=self.tracking.tracks_out,
    #             all_poca=self._all_poca,
    #         )
    #     return self._intersection_coordinates

    @property
    def intersection_coordinates(self) -> List[Tensor]:
        """The intersection points of the track (incoming and outgoing) with the triggered voxels of all muons"""
        if self._intersection_coordinates is None:
            self.process_all_muons()
        return self._intersection_coordinates

    @property
    def triggered_voxels_BETA(self) -> List[Tensor]:
        """The triggered voxels of all muons"""
        if self._triggered_voxels_BETA is None:
            self.process_all_muons()
        return self._triggered_voxels_BETA

    @property
    def path_length_LT(self) -> List[Tensor]:
        """The L and T path length of all muons"""
        if self._path_length_LT is None:
            self.process_all_muons()
        return self._path_length_LT

    def set_M_voxels(self) -> Tensor:
        if self._M_voxels is None:
            self._M_voxels = torch.zeros(self.voi.n_vox_xyz)
        return self._M_voxels

    @property
    def valid_poca(self) -> Tensor:
        """The POCA points inside the volume"""
        return self._valid_poca

    @property
    def W(self) -> Tensor:
        """Weight matrix"""
        if self._W is None or not self._W.all():
            self.process_all_muons()
        return self._W
