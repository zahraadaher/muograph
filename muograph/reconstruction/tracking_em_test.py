from muograph.plotting.voxel import VoxelPlotting
from muograph.plotting.params import configure_plot_theme, font, tracking_figsize
import matplotlib.pyplot as plt
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

"""
Notas mías para entenderme:
    - Los "@..." se llaman 'decoradores', si no me equivoco tienen que ver con que el atrivuto o el método se puede llamar sin necesidad
    de hacer un objeto de esa clase (o algo así)
    - Hay mezcla de lenguaje C++, por ello se suele decar indicado el tipo de variable/método que se tiene para que el código sea más comprensible
    - Hola
"""


class TrackingEM(VoxelPlotting):
    _xyz_in_out_voi: Optional[Tuple[Tensor, Tensor]] = None
    _triggered_voxels: Optional[List[np.ndarray]] = None
    _intersection_coordinates: Optional[Tensor] = None
    _path_length_in_out: Optional[Tuple[Tensor, Tensor]] = None

    _xyz_enters_voi: Optional[Tensor] = None  # to keep
    _xyz_exits_voi: Optional[Tensor] = None  # to keep

    _all_poca: Optional[Tensor] = None

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

    def _compute_intersection_coordinates(
        self,
        voi: Volume,
        triggered_voxels: List[np.ndarray],
        xyz_enters_voi: Tensor,
        xyz_exits_voi: Tensor,
        tracks_in: Tensor,
        tracks_out: Tensor,
        all_poca: Tensor,
    ) -> Tensor:
        """
        A method that retuns the xyz coordinates of the intersection points of the muon track inside the volume with the faces
          of its triggered voxels inside this volume.

        Args:
            voi (Volume): the voxelized volume of interest
            triggered_voxels (List[np.ndarray]): list of the triggered voxels of all muon event of length N_mu
            xyz_enters_voi (Tensor): a tensor with the entry x,y and z coordinates (N_mu, 3)
            xyz_exits_voi (Tensor): a tensor with the exit x,y and z coordinates (N_mu, 3)
            tracks_in (Tensor): the unit direction vector of the entering point (N_mu, 3)
            tracks_out (Tensor): the unit direction vector of the exit point (N_mu, 3)
            all_poca (Tensor): tensor of the xyz positions of all unfiltered poca events (N_mu, 3). If the poca xyz is (0,0,0),
              this coresponds to a filtered event.

        Returns:
            Tensor: tensor of the xyz coordinates of the intersection points of the muon track inside the volume with the faces
          of its triggered voxels inside this volume (N_mu, )
        """
        return torch.tensor([0])

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
        points_in_np = self.tracking.points_in.detach().cpu().numpy()
        points_out_np = self.tracking.points_out.detach().cpu().numpy()
        track_in_np = self.tracking.tracks_in.detach().cpu().numpy()[event]
        track_out_np = self.tracking.tracks_out.detach().cpu().numpy()[event]

        # Y span
        y_span = abs(points_in_np[event, 2] - points_out_np[event, 2])

        fig, ax = plt.subplots(figsize=tracking_figsize)

        # Plot POCA point
        if self.poca is not None:
            if self.all_poca[event, dim_map[proj]["x"]] != 0:  # type: ignore
                ax.scatter(
                    x=self.all_poca[event, dim_map[proj]["x"]],  # type: ignore
                    y=self.all_poca[event, dim_map[proj]["y"]],  # type: ignore
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
                x=self.xyz_enters_voi[event, dim_map[proj]["x"]],  # type: ignore
                y=self.xyz_enters_voi[event, dim_map[proj]["y"]],  # type: ignore
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
                x=self.xyz_exits_voi[event, dim_map[proj]["x"]],  # type: ignore
                y=self.xyz_exits_voi[event, dim_map[proj]["y"]],  # type: ignore
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

        if self.triggered_voxels[event].shape[0] > 0:
            n_trig_vox = f"# triggered voxels = {self.triggered_voxels[event].shape[0]}"
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
        if self.triggered_voxels[event].shape[0] > 0:
            for i, vox_idx in enumerate(self.triggered_voxels[event]):
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

    @property
    def triggered_voxels(self) -> List[np.ndarray]:
        """Voxels triggered by the muon track"""
        if self._triggered_voxels is None:
            self._triggered_voxels = TrackingEM.get_triggered_voxels(
                points_in=self.tracking.points_in,
                points_out=self.tracking.points_out,
                voi=self.voi,
                theta_xy_in=(self.tracking.theta_xy_in[0], self.tracking.theta_xy_in[1]),
                theta_xy_out=(self.tracking.theta_xy_out[0], self.tracking.theta_xy_out[1]),
            )
        return self._triggered_voxels

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
