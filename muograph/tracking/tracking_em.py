import warnings
from typing import List, Tuple, Union
import torch
from torch import Tensor, tensor
import numpy as np
import pandas as pd
import math
from fastprogress import progress_bar
from muograph.volume.volume import Volume
from muograph.reconstruction.poca import POCA
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


class Tracking_EM:
    def __init__(self, voi: Volume, poca_data: POCA, n_events: int = 1000, poca: bool = False, batch_size: int = 100):
        # data:pandas.DataFrame
        self.voi = voi
        self.voxel_edges = voi.voxel_edges
        self.Nvox_Z = voi.n_vox_xyz[2]
        self.n_events = n_events
        self.data = self._extract_data_from_POCA(poca=poca_data, voi=self.voi)
        self.batch_size = batch_size

        if poca_data is None:
            raise ValueError("poca_data is None. Input your POCA class.")
        if poca:
            self.points_in_POCA: Tensor
            self.points_out_POCA: Tensor
            self.points_in_POCA, self.points_out_POCA = self.compute_discrete_tracks(poca=poca)

            self.triggered_voxels_in, self.indices_in, self.ev_hit_vox_count_in = self.find_triggered_voxels(poca=poca, batch_size=self.batch_size)
            self.triggered_voxels_out, self.indices_out, self.ev_hit_vox_count_out = self.find_triggered_voxels(
                _in_=False, poca=poca, batch_size=self.batch_size
            )

            self.path_length_in = np.sqrt(
                (self.data["xyz_in_x"] - self.data["location_x"]) ** 2
                + (self.data["xyz_in_y"] - self.data["location_y"]) ** 2
                + (self.data["xyz_in_z"] - self.data["location_z"]) ** 2
            )
            self.path_length_out = np.sqrt(
                (self.data["location_x"] - self.data["xyz_out_x"]) ** 2
                + (self.data["location_y"] - self.data["xyz_out_y"]) ** 2
                + (self.data["location_z"] - self.data["xyz_out_z"]) ** 2
            )

            self.cos_theta_x_in = (self.data["xyz_in_x"] - self.data["location_x"]) / self.path_length_in
            self.cos_theta_y_in = (self.data["xyz_in_y"] - self.data["location_y"]) / self.path_length_in
            self.cos_theta_z_in = (self.data["xyz_in_z"] - self.data["location_z"]) / self.path_length_in

            self.cos_theta_x_out = (self.data["location_x"] - self.data["xyz_out_x"]) / self.path_length_out
            self.cos_theta_y_out = (self.data["location_y"] - self.data["xyz_out_y"]) / self.path_length_out
            self.cos_theta_z_out = (self.data["location_z"] - self.data["xyz_out_z"]) / self.path_length_out

            self.alpha_x_l_in, self.alpha_x_r_in, self.alpha_y_l_in, self.alpha_y_r_in, self.alpha_z_l_in, self.alpha_z_r_in = self.compute_alpha_vals(
                _in_=True, poca=poca, batch_size=self.batch_size
            )
            self.alpha_x_l_out, self.alpha_x_r_out, self.alpha_y_l_out, self.alpha_y_r_out, self.alpha_z_l_out, self.alpha_z_r_out = self.compute_alpha_vals(
                poca=poca, batch_size=self.batch_size
            )

            self.intersection_coordinates_in = self.compute_intersection_coords(poca=poca, batch_size=self.batch_size)
            self.intersection_coordinates_out = self.compute_intersection_coords(out=True, poca=poca, batch_size=self.batch_size)

            self.triggered_voxels, self.intersection_coordinates, self.indices = self._merge_()

        else:
            self.tracks = self.compute_discrete_tracks()
            self.triggered_voxels, self.indices, self.ev_hit_vox_count = self.find_triggered_voxels(poca=poca)
            self.path_length = np.sqrt(
                (self.data["xyz_in_x"] - self.data["xyz_out_x"]) ** 2
                + (self.data["xyz_in_y"] - self.data["xyz_out_y"]) ** 2
                + (self.data["xyz_in_z"] - self.data["xyz_out_z"]) ** 2
            )
            self.cos_theta_x = (self.data["xyz_in_x"] - self.data["xyz_out_x"]) / self.path_length
            self.cos_theta_y = (self.data["xyz_in_y"] - self.data["xyz_out_y"]) / self.path_length
            self.cos_theta_z = (self.data["xyz_in_z"] - self.data["xyz_out_z"]) / self.path_length
            self.alpha_x_l, self.alpha_x_r, self.alpha_y_l, self.alpha_y_r, self.alpha_z_l, self.alpha_z_r = self.compute_alpha_vals()
            self.intersection_coordinates = self.compute_intersection_coords()

        self.W, self.M, self.Path_Length, self.T2, self.Hit = self.calculate_path_length()
        self.Dx, self.Dy = self.compute_observed_data()

    def _compute_xyz_in_out(
        self,
        points_in: Tensor,
        points_out: Tensor,
        voi: Volume,
        theta_xy_in: Tuple[Tensor, Tensor],
        theta_xy_out: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Compute muon position (x,y,z) when enters/exits the volume,
        both for the incoming and outgoing tracks.

        Args:
            - points_in (Tensor): Points on incoming muon tracks.
            - points_out (Tensor): Points on outgoing muon tracks.
            - voi (Volume): Instance of the volume class.
            - theta_xy_in (Tensor): The incoming projected zenith angle in XZ and YZ plane.
            - theta_xy_out (Tensor): The outgoing projected zenith angle in XZ and YZ plane.

        Returns:
            - xyz_in_VOI [3, 2, mu] the muon position when entering/exiting the volume,
            for the INCOMING tracks.
            - xyz_out_VOI [3, 2, mu] the muon position when entering/exiting the volume,
            for the OUTGOING tracks.
        """
        n_mu = theta_xy_in[0].size(0)
        xyz_in_voi, xyz_out_voi = torch.zeros((n_mu, 2, 3)), torch.zeros((n_mu, 2, 3))

        for point, theta_xy, pm, xyz in zip(
            [points_in, points_out],
            [theta_xy_in, theta_xy_out],
            [1, -1],
            [xyz_in_voi, xyz_out_voi],
        ):
            dz = (abs(point[:, 2] - voi.xyz_max[2]), abs(point[:, 2] - voi.xyz_min[2]))

            for coord, theta in zip([0, 1], theta_xy):
                xyz[:, 0, coord] = point[:, coord] - dz[1] * torch.tan(theta) * (pm)
                xyz[:, 1, coord] = point[:, coord] - dz[0] * torch.tan(theta) * (pm)
            xyz[:, 0, 2], xyz[:, 1, 2] = voi.xyz_max[2], voi.xyz_min[2]

        return xyz_in_voi[:, 0, :], xyz_out_voi[:, 1, :]

    def _extract_data_from_POCA(self, poca: POCA, voi: Volume) -> pd.DataFrame:
        theta_x_in = poca.tracks.theta_xy_in[0]
        theta_y_in = poca.tracks.theta_xy_in[1]

        theta_x_out = poca.tracks.theta_xy_out[0]
        theta_y_out = poca.tracks.theta_xy_out[1]

        poca_x = poca.poca_points[:, 0]
        poca_y = poca.poca_points[:, 1]
        poca_z = poca.poca_points[:, 2]

        mom = torch.sqrt(2 * 105.7 * poca.tracks.E[:]) * 1e-3  # mom in GeV

        inn, out = self._compute_xyz_in_out(
            points_in=poca.tracks.points_in,
            points_out=poca.tracks.points_out,
            voi=voi,
            theta_xy_in=(theta_x_in, theta_y_in),
            theta_xy_out=(theta_x_out, theta_y_out),
        )

        z_in = inn[:, 2]
        z_out = out[:, 2]
        x_in = inn[:, 0]
        y_in = inn[:, 1]
        x_out = out[:, 0]
        y_out = out[:, 1]

        data = pd.DataFrame(
            {
                "xyz_in_x": x_in,
                "xyz_in_y": y_in,
                "xyz_in_z": z_in,
                "xyz_out_x": x_out,
                "xyz_out_y": y_out,
                "xyz_out_z": z_out,
                "theta_in_x": theta_x_in,
                "theta_in_y": theta_y_in,
                "theta_out_x": theta_x_out,
                "theta_out_y": theta_y_out,
                "location_x": poca_x,
                "location_y": poca_y,
                "location_z": poca_z,
                "mom": mom,
            }
        )
        return data[: self.n_events]

    def compute_discrete_tracks(self, poca: bool = False) -> Union[Tuple[Tensor, Tensor], Tensor]:
        """
        Function computes x,y,z position at Zmax and Zmin of each voxel layer (for incoming and outgoing tracks)

        Outputs:
                two tensors/lists (in case of PoCA) having x,y,z position at Zmax and Zmin of each voxel
                layer (for incoming and outgoing tracks), size = [coordinate,Nlayer_along_Z + 1, Nevents]

        """

        if poca:
            k = np.linspace(0, 1, num=100)
            points_in_POCA = np.stack(
                [
                    self.data["xyz_in_x"][:, np.newaxis] + (self.data["location_x"] - self.data["xyz_in_x"])[:, np.newaxis] * k,
                    self.data["xyz_in_y"][:, np.newaxis] + (self.data["location_y"] - self.data["xyz_in_y"])[:, np.newaxis] * k,
                    self.data["xyz_in_z"][:, np.newaxis] + (self.data["location_z"] - self.data["xyz_in_z"])[:, np.newaxis] * k,
                ],
                axis=-1,
            )

            points_out_POCA = np.stack(
                [
                    self.data["location_x"][:, np.newaxis] + (self.data["xyz_out_x"] - self.data["location_x"])[:, np.newaxis] * k,
                    self.data["location_y"][:, np.newaxis] + (self.data["xyz_out_y"] - self.data["location_y"])[:, np.newaxis] * k,
                    self.data["location_z"][:, np.newaxis] + (self.data["xyz_out_z"] - self.data["location_z"])[:, np.newaxis] * k,
                ],
                axis=-1,
            )

            return torch.Tensor(points_in_POCA), torch.Tensor(points_out_POCA)

        else:
            voxels_edges = self.voxel_edges

            xyz_in_x = self.data["xyz_in_x"]
            xyz_in_y = self.data["xyz_in_y"]
            xyz_in_z = self.data["xyz_in_z"]

            xyz_out_x = self.data["xyz_out_x"]
            xyz_out_y = self.data["xyz_out_y"]
            xyz_out_z = self.data["xyz_out_z"]

            dx = xyz_out_x - xyz_in_x
            dy = xyz_out_y - xyz_in_y
            dz = xyz_out_z - xyz_in_z

            theta_x = torch.arctan(tensor(dx / dz))
            theta_y = torch.arctan(tensor(dy / dz))

            Z_discrete = torch.linspace(torch.min(voxels_edges[:, :, :, :, 2]).item(), torch.max(voxels_edges[:, :, :, :, 2]).item(), self.Nvox_Z + 1)
            Z_discrete.unsqueeze_(1)
            Z_discrete = torch.round(Z_discrete.expand(len(Z_discrete), len(self.data)), decimals=3)
            x_track = tensor(xyz_in_x) + (xyz_out_z[0] - Z_discrete) * torch.tan(theta_x)
            y_track = tensor(xyz_in_y) + (xyz_out_z[0] - Z_discrete) * torch.tan(theta_y)

            return torch.stack([x_track, y_track, Z_discrete.flip(dims=(0,))])

    # def find_triggered_voxels(self,_in_=True,poca=False):

    #     '''
    #     Function identifies the list of voxels that were triggered by the particle, along with
    #     their respective indices in the input data file.

    #     Returns:

    #         triggered_voxels: A list of lists of tuples, where each tuple represents a voxel.
    #                           The list contains indices of all the voxels that were triggered
    #                           by the particle.

    #         vox_loc: A list of the indices of events where the triggered voxels were detected.
    #     '''

    #     triggered_voxels = []
    #     vox_loc = []
    #     ev_hit_vox_count=torch.zeros(len(self.data))

    #     if not poca:
    #         zmax,zmin=self.voi.xyz_max[2], self.voi.xyz_min[2]

    #     print('Finding triggered voxels.')

    #     for ev in progress_bar(range(len(self.data))):
    #         edge = self.voi.voxel_edges

    #         if _in_ and poca:

    #             x1,y1=self.data['xyz_in_x'][ev],self.data['xyz_in_y'][ev]
    #             x2,y2=self.data['location_x'][ev],self.data['location_y'][ev]
    #             IN = self.points_in_POCA[ev][0]
    #             OUT = self.points_in_POCA[ev][-1]
    #             zmax,zmin=self.data['xyz_in_z'][ev], self.data['location_z'][ev]

    #         elif poca:

    #             x1,y1=self.data['location_x'][ev],self.data['location_y'][ev]
    #             x2,y2=self.data['xyz_out_x'][ev],self.data['xyz_out_y'][ev]
    #             IN = self.points_out_POCA[ev][0]
    #             OUT = self.points_out_POCA[ev][-1]
    #             zmax,zmin=self.data['location_z'][ev], self.data['xyz_out_z'][ev]

    #         else:

    #             x1,y1,z1=self.data['xyz_in_x'][ev],self.data['xyz_in_y'][ev],self.data['xyz_in_z'][ev]
    #             x2,y2,z2=self.data['xyz_out_x'][ev],self.data['xyz_out_y'][ev],self.data['xyz_out_z'][ev]
    #             IN = self.tracks[:, 0, ev]
    #             OUT = self.tracks[:, -1, ev]

    #         theta_x = math.atan((OUT[0] - IN[0]) / (OUT[2] - IN[2]))
    #         theta_y = math.atan((OUT[1] - IN[1]) / (OUT[2] - IN[2]))

    #         z = torch.linspace(zmax, zmin, 100)
    #         x = IN[0] + (z - IN[2]) * math.tan(theta_x)
    #         y = IN[1] + (z - IN[2]) * math.tan(theta_y)

    #         xyz = torch.zeros(100, 3)
    #         xyz[:, 0] = x
    #         xyz[:, 1] = y
    #         xyz[:, 2] = z

    #         edge = edge[:, :, :, :, None, :]
    #         edge = edge.expand(self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2], 2, len(xyz), 3)
    #         mask = ((edge[:, :, :, 0, :, 0] < xyz[:, 0]) & (edge[:, :, :, 1, :, 0] > xyz[:, 0]) &
    #                 (edge[:, :, :, 0, :, 1] < xyz[:, 1]) & (edge[:, :, :, 1, :, 1] > xyz[:, 1]) &
    #                 (edge[:, :, :, 0, :, 2] < xyz[:, 2]) & (edge[:, :, :, 1, :, 2] > xyz[:, 2]))

    #         indices = torch.nonzero(mask, as_tuple=False)
    #         indices = indices[:, :-1].unique(dim=0).tolist()

    #         sign_x, sign_y = (-1, -1) if x1 > x2 and y1 > y2 else \
    #                          (-1,  1) if x1 > x2 and y1 < y2 else \
    #                          ( 1, -1) if x1 < x2 and y1 > y2 else \
    #                          ( 1,  1)
    #         key = lambda k: (k[2], sign_x * k[0], sign_y * k[1])

    #         indices = sorted(indices, key=key)

    #         if len(indices)>0:
    #             pair = (indices[0], indices[-1])
    #             if pair[0][1:] == pair[1][1:] and ((pair[0][0] > pair[1][0] and indices[0][0] < indices[-1][0]) or (pair[0][0] < pair[1][0] and indices[0][0] > indices[-1][0])):
    #                 indices.reverse()

    #         triggered_voxels.append(indices)
    #         ev_hit_vox_count[ev]=len(indices)
    #         vox_loc.append(ev)

    #     return triggered_voxels, vox_loc, ev_hit_vox_count

    def find_triggered_voxels(self, _in_: bool = True, poca: bool = False, batch_size: int = 100) -> Tuple[List, List, Tensor]:
        """
        Batch version of find_triggered_voxels function to avoid memory overload.
        """
        triggered_voxels = []
        vox_loc = []
        ev_hit_vox_count = torch.zeros(len(self.data))

        if not poca:
            zmax, zmin = self.voi.xyz_max[2], self.voi.xyz_min[2]

        # Create a DataLoader for batching the events
        dataset = TensorDataset(torch.arange(len(self.data)))  # <-- NEW LINE
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)  # <-- NEW LINE

        print("Finding triggered voxels in batches.")

        for batch in progress_bar(dataloader):  # <-- NEW LINE
            batch_indices = batch[0].tolist()  # Extract batch indices  # <-- NEW LINE

            for ev in batch_indices:  # <-- UPDATED TO USE BATCHED DATA
                edge = self.voi.voxel_edges

                if _in_ and poca:
                    x1, y1 = self.data["xyz_in_x"][ev], self.data["xyz_in_y"][ev]
                    x2, y2 = self.data["location_x"][ev], self.data["location_y"][ev]
                    IN = self.points_in_POCA[ev][0]
                    OUT = self.points_in_POCA[ev][-1]
                    zmax, zmin = self.data["xyz_in_z"][ev], self.data["location_z"][ev]

                elif poca:
                    x1, y1 = self.data["location_x"][ev], self.data["location_y"][ev]
                    x2, y2 = self.data["xyz_out_x"][ev], self.data["xyz_out_y"][ev]
                    IN = self.points_out_POCA[ev][0]
                    OUT = self.points_out_POCA[ev][-1]
                    zmax, zmin = self.data["location_z"][ev], self.data["xyz_out_z"][ev]

                else:
                    if isinstance(self.tracks, Tensor):
                        x1, y1 = self.data["xyz_in_x"][ev], self.data["xyz_in_y"][ev]  # , self.data["xyz_in_z"][ev]
                        x2, y2 = self.data["xyz_out_x"][ev], self.data["xyz_out_y"][ev]  # , self.data["xyz_out_z"][ev]
                        IN = self.tracks[:, 0, ev]
                        OUT = self.tracks[:, -1, ev]

                    else:
                        raise TypeError(f"Expected a Tensor or a tuple of Tensors, but got {type(IN)}.")

                theta_x = math.atan((OUT[0] - IN[0]) / (OUT[2] - IN[2]))
                theta_y = math.atan((OUT[1] - IN[1]) / (OUT[2] - IN[2]))

                z = torch.linspace(zmax, zmin, 100)  # type: ignore
                x = IN[0] + (z - IN[2]) * math.tan(theta_x)
                y = IN[1] + (z - IN[2]) * math.tan(theta_y)

                xyz = torch.zeros(100, 3)
                xyz[:, 0] = x
                xyz[:, 1] = y
                xyz[:, 2] = z

                edge = edge[:, :, :, :, None, :]
                edge = edge.expand(self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2], 2, len(xyz), 3)
                mask = (
                    (edge[:, :, :, 0, :, 0] < xyz[:, 0])
                    & (edge[:, :, :, 1, :, 0] > xyz[:, 0])
                    & (edge[:, :, :, 0, :, 1] < xyz[:, 1])
                    & (edge[:, :, :, 1, :, 1] > xyz[:, 1])
                    & (edge[:, :, :, 0, :, 2] < xyz[:, 2])
                    & (edge[:, :, :, 1, :, 2] > xyz[:, 2])
                )

                indices = torch.nonzero(mask, as_tuple=False)
                indices = indices[:, :-1].unique(dim=0).tolist()

                # sign_x, sign_y = (-1, -1) if x1 > x2 and y1 > y2 else (-1, 1) if x1 > x2 and y1 < y2 else (1, -1) if x1 < x2 and y1 > y2 else (1, 1)
                # key = lambda k: (k[2], sign_x * k[0], sign_y * k[1])

                def key_func(k: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
                    return (k[2], sign_x * k[0], sign_y * k[1])

                sign_x, sign_y = (-1, -1) if x1 > x2 and y1 > y2 else (-1, 1) if x1 > x2 and y1 < y2 else (1, -1) if x1 < x2 and y1 > y2 else (1, 1)
                key = key_func

                indices = torch.tensor(sorted(indices, key=key))

                if len(indices) > 0:
                    pair = (indices[0], indices[-1])
                    if pair[0][1:] == pair[1][1:] and (
                        (pair[0][0] > pair[1][0] and indices[0][0] < indices[-1][0]) or (pair[0][0] < pair[1][0] and indices[0][0] > indices[-1][0])
                    ):
                        indices.flip(dims=(0,))

                triggered_voxels.append(indices)
                ev_hit_vox_count[ev] = len(indices)
                vox_loc.append(ev)

        return triggered_voxels, vox_loc, ev_hit_vox_count

    def compute_alpha_vals(self, _in_: bool = False, poca: bool = False, batch_size: int = 100) -> Tuple[List, List, List, List, List, List]:
        """
        Function computes the alpha values for all voxels triggered by muons in batches.

        The alpha parameter is used to calculate the attenuation or absorption of the muon
        as it passes through a material or tissue. The function now processes the events in
        batches for memory efficiency.

        Outputs:
            alpha_x_l, alpha_x_r,
            alpha_y_l, alpha_y_r,
            alpha_z_l, alpha_z_r: Each list contains sublists, which represent alpha values
                                    calculated from the input data for different voxels.
        """

        alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l, alpha_z_r = [], [], [], [], [], []

        if _in_ and poca:
            triggered_voxels = self.triggered_voxels_in
            indices = self.indices_in
            cos_theta_x, cos_theta_y, cos_theta_z = self.cos_theta_x_in, self.cos_theta_y_in, self.cos_theta_z_in

        elif poca:
            triggered_voxels = self.triggered_voxels_out
            indices = self.indices_out
            cos_theta_x, cos_theta_y, cos_theta_z = self.cos_theta_x_out, self.cos_theta_y_out, self.cos_theta_z_out

        else:
            triggered_voxels = self.triggered_voxels
            indices = self.indices
            cos_theta_x, cos_theta_y, cos_theta_z = self.cos_theta_x, self.cos_theta_y, self.cos_theta_z

        # Prepare batching
        dataset = TensorDataset(torch.arange(len(triggered_voxels)))  # Event indices
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print("Computing alpha vals in batches.")

        for batch in dataloader:
            batch_indices = batch[0].tolist()

            for i in batch_indices:
                alpha_x_l_i, alpha_x_r_i, alpha_y_l_i, alpha_y_r_i, alpha_z_l_i, alpha_z_r_i = [], [], [], [], [], []

                if len(triggered_voxels[i]) > 0:
                    if _in_ and poca:
                        x_in, y_in, z_in = (
                            self.data.iloc[indices[i]]["xyz_in_x"],
                            self.data.iloc[indices[i]]["xyz_in_y"],
                            self.data.iloc[indices[i]]["xyz_in_z"],
                        )
                    elif poca:
                        x_in, y_in, z_in = (
                            self.data.iloc[indices[i]]["location_x"],
                            self.data.iloc[indices[i]]["location_y"],
                            self.data.iloc[indices[i]]["location_z"],
                        )
                    else:
                        x_in, y_in, z_in = (
                            self.data.iloc[indices[i]]["xyz_in_x"],
                            self.data.iloc[indices[i]]["xyz_in_y"],
                            self.data.iloc[indices[i]]["xyz_in_z"],
                        )

                    for j in range(len(triggered_voxels[i])):
                        x, y, z = triggered_voxels[i][j][0], triggered_voxels[i][j][1], triggered_voxels[i][j][2]

                        alpha_l = (self.voi.voxel_edges[x, y, z, 0, 0] - x_in) / cos_theta_x.iloc[indices[i]]
                        alpha_r = (self.voi.voxel_edges[x, y, z, 1, 0] - x_in) / cos_theta_x.iloc[indices[i]]

                        alpha_x_l_i.append(alpha_l)
                        alpha_x_r_i.append(alpha_r)

                        alpha_l = (self.voi.voxel_edges[x, y, z, 0, 1] - y_in) / cos_theta_y.iloc[indices[i]]
                        alpha_r = (self.voi.voxel_edges[x, y, z, 1, 1] - y_in) / cos_theta_y.iloc[indices[i]]

                        alpha_y_l_i.append(alpha_l)
                        alpha_y_r_i.append(alpha_r)

                        alpha_l = (self.voi.voxel_edges[x, y, z, 0, 2] - z_in) / cos_theta_z.iloc[indices[i]]
                        alpha_r = (self.voi.voxel_edges[x, y, z, 1, 2] - z_in) / cos_theta_z.iloc[indices[i]]

                        alpha_z_l_i.append(alpha_l)
                        alpha_z_r_i.append(alpha_r)

                alpha_x_l.append(alpha_x_l_i)
                alpha_x_r.append(alpha_x_r_i)
                alpha_y_l.append(alpha_y_l_i)
                alpha_y_r.append(alpha_y_r_i)
                alpha_z_l.append(alpha_z_l_i)
                alpha_z_r.append(alpha_z_r_i)

        return alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l, alpha_z_r

    def compute_intersection_coords(self, out: bool = False, poca: bool = False, batch_size: int = 100) -> List:
        """
        Function calculates the intersection points between voxels and muon trajectories.

        Outputs:
                coordinates: list of coordinates that describe the trajectory of each particle
                            with shape (num_events,).
        """

        # def max_val(lst, index):
        #     return max(enumerate(sub[index] for sub in lst), key=itemgetter(1))

        def trajectory(alpha: Tensor, index: Tensor, out: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            location = ("xyz_in_x", "xyz_in_y", "xyz_in_z") if not out else ("location_x", "location_y", "location_z")
            data = self.data.iloc[index]

            if poca:
                x = data[location[0]] + (self.cos_theta_x_in.iloc[index] if not out else self.cos_theta_x_out.iloc[index]) * alpha.numpy()
                y = data[location[1]] + (self.cos_theta_y_in.iloc[index] if not out else self.cos_theta_y_out.iloc[index]) * alpha.numpy()
                z = data[location[2]] + (self.cos_theta_z_in.iloc[index] if not out else self.cos_theta_z_out.iloc[index]) * alpha.numpy()
            else:
                x = data[location[0]] + self.cos_theta_x.iloc[index] * alpha.numpy()
                y = data[location[1]] + self.cos_theta_y.iloc[index] * alpha.numpy()
                z = data[location[2]] + self.cos_theta_z.iloc[index] * alpha.numpy()

            # x = 0 if x < 0 else x
            # y = 0 if y < 0 else y

            return x, y, z

        coordinates = []

        # Set the triggered voxels and alpha values depending on whether `poca` and `out` are set
        if poca:
            triggered_voxels = self.triggered_voxels_out if out else self.triggered_voxels_in
            alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l = (
                self.alpha_x_l_out if out else self.alpha_x_l_in,
                self.alpha_x_r_out if out else self.alpha_x_r_in,
                self.alpha_y_l_out if out else self.alpha_y_l_in,
                self.alpha_y_r_out if out else self.alpha_y_r_in,
                self.alpha_z_l_out if out else self.alpha_z_l_in,
            )
        else:
            triggered_voxels = self.triggered_voxels
            alpha_x_l, alpha_x_r, alpha_y_l, alpha_y_r, alpha_z_l = (
                self.alpha_x_l,
                self.alpha_x_r,
                self.alpha_y_l,
                self.alpha_y_r,
                self.alpha_z_l,
            )

        xmax, ymax, zmax = self.voi.xyz_max
        xmin, ymin, zmin = self.voi.xyz_min
        print("Computing voxels and muon trajectories intersection coordinates.")

        # Process events in batches
        for ev_batch in range(0, len(triggered_voxels), batch_size):
            # Slice the current batch of triggered_voxels and alpha values
            ev_batch_voxels = triggered_voxels[ev_batch : ev_batch + batch_size]
            ev_batch_alpha_x_l = alpha_x_l[ev_batch : ev_batch + batch_size]
            ev_batch_alpha_x_r = alpha_x_r[ev_batch : ev_batch + batch_size]
            ev_batch_alpha_y_l = alpha_y_l[ev_batch : ev_batch + batch_size]
            ev_batch_alpha_y_r = alpha_y_r[ev_batch : ev_batch + batch_size]
            ev_batch_alpha_z_l = alpha_z_l[ev_batch : ev_batch + batch_size]

            for ev, triggered_voxel in enumerate(ev_batch_voxels):
                if poca:
                    ev_indx = self.indices_out[ev + ev_batch] if out else self.indices_in[ev + ev_batch]

                else:
                    ev_indx = self.indices[ev + ev_batch]

                # ev_indx = self.indices_out[ev_batch + ev] if out else self.indices_in[ev_batch + ev]
                data = self.data.iloc[ev_indx]

                if poca:
                    # Fetch appropriate coordinates for poco events
                    x_in, x_out = data[["location_x", "xyz_out_x"]] if out else data[["xyz_in_x", "location_x"]]
                    y_in, y_out = data[["location_y", "xyz_out_y"]] if out else data[["xyz_in_y", "location_y"]]
                    z_in, z_out = data[["location_z", "xyz_out_z"]] if out else data[["xyz_in_z", "location_z"]]
                else:
                    # For regular events
                    x_in, x_out = data[["xyz_in_x", "xyz_out_x"]]
                    y_in, y_out = data[["xyz_in_y", "xyz_out_y"]]
                    z_in, z_out = data[["xyz_in_z", "xyz_out_z"]]

                # Compute trajectory coordinates for the voxel hits in the event
                # ev_coord = [(0, 0, 0)] if len(triggered_voxel) > 0 else []
                ev_coord = [(np.array(0), np.array(0), np.array(0))] if len(triggered_voxel) > 0 else []

                for hit_vox in range(len(triggered_voxel) - 1):
                    curr_vox_x, curr_vox_y, curr_vox_z = triggered_voxel[hit_vox]
                    next_vox_x, next_vox_y, next_vox_z = triggered_voxel[hit_vox + 1]

                    if curr_vox_z != next_vox_z:
                        coord = trajectory(ev_batch_alpha_z_l[ev][hit_vox + 1], ev_indx, out)
                    elif curr_vox_y > next_vox_y:
                        coord = trajectory(ev_batch_alpha_y_l[ev][hit_vox], ev_indx, out)
                    elif curr_vox_y < next_vox_y:
                        coord = trajectory(ev_batch_alpha_y_r[ev][hit_vox], ev_indx, out)
                    elif curr_vox_x > next_vox_x:
                        coord = trajectory(ev_batch_alpha_x_l[ev][hit_vox], ev_indx, out)
                    else:
                        coord = trajectory(ev_batch_alpha_x_r[ev][hit_vox], ev_indx, out)

                    if coord not in ev_coord and (zmin <= coord[2] <= zmax):
                        ev_coord.append(coord)

                ev_coord.append((np.array(0), np.array(0), np.array(0)))

                # Handle corner cases for event coordinates
                if len(triggered_voxel) > 0:
                    if xmin <= x_in <= xmax and ymin <= y_in <= ymax:
                        ev_coord[0] = (x_in, y_in, z_in)

                    if xmin <= x_out <= xmax and ymin <= y_out <= ymax:
                        ev_coord[-1] = (x_out, y_out, z_out)

                coordinates.append(ev_coord)

        return coordinates

    def _merge_(self) -> Tuple[List, List, List]:
        vox_in, vox_out = self.triggered_voxels_in, self.triggered_voxels_out
        coord_in, coord_out = self.intersection_coordinates_in, self.intersection_coordinates_out

        triggered_voxels, indices, coordinates = [], [], []

        for i, (x1, y1, x2, y2, coords_in, coords_out, vox_in_i, vox_out_i) in enumerate(
            zip(self.data["xyz_in_x"], self.data["xyz_in_y"], self.data["xyz_out_x"], self.data["xyz_out_y"], coord_in, coord_out, vox_in, vox_out)
        ):
            # Skip cases with fewer coordinates
            if len(coords_in) <= 1:
                continue

            # Merge voxel and coordinate lists
            voxels = set(map(tuple, vox_in_i + vox_out_i))
            coords = set(map(tuple, coords_in + coords_out))

            indices.append(i)

            # Determine sorting signs
            sign_x, sign_y = (-1, -1) if x1 > x2 and y1 > y2 else (-1, 1) if x1 > x2 and y1 < y2 else (1, -1) if x1 < x2 and y1 > y2 else (1, 1)

            # Apply sorted transformations
            triggered_voxels.append(sorted(voxels, key=lambda k: (k[2], sign_x * k[0], sign_y * k[1])))  # type: ignore
            coordinates.append(sorted(coords, key=lambda k: (-k[2], sign_x * k[0], sign_y * k[1])))  # type: ignore

        return triggered_voxels, coordinates, indices

    def calculate_path_length(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Function calculates the path lengths between intersections along the path taken by muon through a medium.

        Outputs:

            W: a torch tensor of shape (len(self.triggered_voxels), N, N, N1, 2, 2),
                where N, N1 are the dimensions of the voxel grid. It contains the path integrals
                for each voxel and each interval in the 2x2 matrix format.

            M: a torch tensor of shape (N, N, N1), containing the number of path integrals that
                pass through each voxel.

            L: a torch tensor of shape (len(self.triggered_voxels), N, N, N1)
                         containing the length of the muon path in each voxel.

            T: a torch tensor of shape (len(self.triggered_voxels), N, N, N1) containing the
                accumulated length of the muon path from the end of each voxel to the end of the path.

            Hit: a torch tensor of shape (len(self.triggered_voxels), N, N, N1), containing boolean values
                  indicating which voxels are crossed by the muon path.

        """

        def compute_path_length(coordinates: List) -> List:
            path_len = []
            for coords_arr in coordinates:
                coords_arr = np.array(coords_arr)
                diff = np.diff(coords_arr, axis=0)
                if len(diff) > 0:
                    path_len.append(np.sqrt(np.sum(diff**2, axis=1)))
            return path_len

        path_len = compute_path_length(self.intersection_coordinates)
        N, Nj, N1 = self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2]
        M = torch.zeros(self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        Hit = torch.zeros(len(self.triggered_voxels), self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        L = torch.zeros(len(self.triggered_voxels), self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        T = torch.zeros(len(self.triggered_voxels), self.voi.n_vox_xyz[0], self.voi.n_vox_xyz[1], self.voi.n_vox_xyz[2])
        W = torch.zeros(len(self.triggered_voxels), N, Nj, N1, 2, 2)
        print("Computing inputs for the EM steps: (W, L, T and M)")
        for i in progress_bar(range(len(self.triggered_voxels))):
            voxels = self.triggered_voxels[i]
            if i < len(path_len):
                lrev = list(reversed(path_len[i]))
                lrev_cum_sum = torch.cumsum(torch.tensor(lrev), dim=0)
                lrev_cum_sum_rev = list(reversed(lrev_cum_sum))
                lrev_cum_sum_rev = lrev_cum_sum_rev[1:]
                lrev_cum_sum_rev.append(path_len[i][-1])
                lrev_cum_sum_rev.append(0)

                for j in range(len(voxels)):
                    vox = voxels[j]
                    if j < len(path_len[i]):
                        L[i, vox[0], vox[1], vox[2]] = torch.from_numpy(np.array([path_len[i][j]]))
                    if j < len(lrev_cum_sum_rev):
                        T[i, vox[0], vox[1], vox[2]] = torch.from_numpy(np.array([lrev_cum_sum_rev[j]]))

            W[i, :, :, :, 0, 0] = L[i, :, :, :]
            W[i, :, :, :, 0, 1] = (L[i, :, :, :] ** 2) / 2 + L[i, :, :, :] * T[i, :, :, :]
            W[i, :, :, :, 1, 0] = (L[i, :, :, :] ** 2) / 2 + L[i, :, :, :] * T[i, :, :, :]
            W[i, :, :, :, 1, 1] = (L[i, :, :, :] ** 3) / 3 + (L[i, :, :, :] ** 2) * T[i, :, :, :] + L[i, :, :, :] * (T[i, :, :, :] ** 2)

            Hit[i, :, :, :] = L[i, :, :, :] != 0
            M[:, :, :] += Hit[i, :, :, :]

        return W, M, L, T, Hit

    def compute_observed_data(self) -> Tuple[Tensor, Tensor]:
        """
        Function computes the differences in thetas and path lengths in x and y directions, respectively,
        for voxels triggered by muons.

        Outputs:
            Dx and Dy: tensors with shape (len(self.triggered_voxels), 2), representing the calculated
                       differences in thetas and path lengths in x and y directions, respectively.
        """

        Dy = torch.zeros(len(self.triggered_voxels), 2)
        Dx = torch.zeros(len(self.triggered_voxels), 2)
        indx_count = 0
        print("Computing inputs for the EM steps: (Dx and Dx)")
        for i in self.indices:
            x0, x1, y0, y1, z0, z1 = (
                self.data["xyz_in_x"][i],
                self.data["xyz_out_x"][i],
                self.data["xyz_in_y"][i],
                self.data["xyz_out_y"][i],
                self.data["xyz_in_z"][i],
                self.data["xyz_out_z"][i],
            )
            theta_x0, theta_x1, theta_y0, theta_y1 = (
                self.data["theta_in_x"][i],
                self.data["theta_out_x"][i],
                self.data["theta_in_y"][i],
                self.data["theta_out_y"][i],
            )

            deltathetax = theta_x1 - theta_x0
            deltathetay = theta_y1 - theta_y0

            Lxy = np.sqrt(1 + np.tan(theta_x0) ** 2 + np.tan(theta_y0) ** 2)

            xp = x0 + np.tan(theta_x0) * abs(z0 - z1)
            yp = y0 + np.tan(theta_y0) * abs(z0 - z1)

            # deltatheta_x_comp=((x1 - xp)*( Lxy / np.sqrt(1 + np.tan(theta_x0)**2))*np.cos((theta_x1 + theta_x0)/2))
            # deltatheta_y_comp=((y1 - yp)*( Lxy / np.sqrt(1 + np.tan(theta_y0)**2))*np.cos((theta_y1 + theta_y0)/2))

            deltatheta_x_comp = (x1 - xp) * Lxy * np.cos(theta_x0) * np.cos(theta_x1) / np.cos(deltathetax)
            deltatheta_y_comp = (y1 - yp) * Lxy * np.cos(theta_y0) * np.cos(theta_y1) / np.cos(deltathetay)

            # revert this

            Dx[indx_count, 0] = torch.from_numpy(np.array([deltathetax]))
            Dy[indx_count, 0] = torch.from_numpy(np.array([deltathetay]))

            Dx[indx_count, 1] = torch.from_numpy(np.array([deltatheta_x_comp]))
            Dy[indx_count, 1] = torch.from_numpy(np.array([deltatheta_y_comp]))

            indx_count = indx_count + 1

        return Dx, Dy
