from hits.hits import Hits
from tracking.tracking import Tracking, TrackingMST
from reconstruction.poca import POCA
from volume.volume import Volume
import os

# Test data file path
test_hit_file = os.path.dirname(__file__) + "/../data/iron_barrel/barrel_and_cubes_scattering.csv"


def get_mst(hits_file: str) -> TrackingMST:
    hits_in = Hits(
        plane_labels=(0, 1, 2),
        csv_filename=test_hit_file,
        spatial_res=[1.0, 1.0, 0.0],
        energy_range=[0.0, 1_000_000],
        efficiency=0.98,
        input_unit="mm",
    )

    hits_out = Hits(
        plane_labels=(3, 4, 5),
        csv_filename=test_hit_file,
        spatial_res=[1.0, 1.0, 0.0],
        energy_range=[0.0, 1_000_000],
        efficiency=0.98,
        input_unit="mm",
    )

    tracks_in = Tracking(label="above", hits=hits_in)
    tracks_out = Tracking(label="below", hits=hits_out)

    return TrackingMST(trackings=(tracks_in, tracks_out))


def test_poca_predictions() -> None:
    mst = get_mst(test_hit_file)

    voi = Volume(position=[0, 0, -1200], dimension=[1000, 600, 600], voxel_width=20)

    poca = POCA(mst, voi=voi)

    n_poca_uranium_x_region = poca.n_poca_per_vox[23:28].float().mean()
    n_poca_empty_x_region = poca.n_poca_per_vox[-5:].float().mean()

    assert (
        n_poca_uranium_x_region > n_poca_empty_x_region
    ), "The average number of POCA points per voxel in the uranium x region {n_poca_uranium_x_region} must be higher than in the empty region {n_poca_empty_x_region}"
