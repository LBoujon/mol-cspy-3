from ase.io.trajectory import Trajectory
from n2p2_ase_calculator import ase_atoms_to_n2p2


def poscar_from_trajectory(traj_fname: str, snapshot_number: int, output_folder='.', kspacing=0.05, ncpu=40):
    trajectory = Trajectory(traj_fname)

    # convert atoms object to Snapshot
    s = ase_atoms_to_n2p2(trajectory[snapshot_number], output_format='snapshot')
    s.generate_vasp_inputs(
            output_folder=f'{output_folder}',
            kspacing=kspacing,
            ncpu=ncpu,
        )
