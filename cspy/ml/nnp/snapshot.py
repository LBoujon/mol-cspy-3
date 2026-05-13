import numpy as np
from collections import OrderedDict
import os
from math import gcd
from subprocess import call
from .units import *
from .cp2k_to_res2 import xyz_to_res
from typing import Union, List, Optional, Dict, Tuple
from atom import Atom

if os.path.exists("/path/to/potpaw_PBE"):
    potcar_path = "/path/to/potpaw_PBE"
else:
    raise Exception("POTCAR path not found. Please set the correct path to POTCAR files.")


class Snapshot:

    def __init__(
        self,
        label: str,
        atoms: List[Atom],
        energy: float,
        box_bounds: Optional[np.ndarray] = None,
        lattice_vectors: Optional[np.ndarray] = None
    ) -> None:
        """
        Initialize a Snapshot object.

        Parameters
        ----------
        label: str
            Name of the snapshot.
        atoms: List[Atom]
            List of Atom objects representing the atoms in the snapshot.
        energy: float
            Total energy of the snapshot.
        box_bounds: Optional[np.ndarray]
            Box bounds of the snapshot, if applicable. If not provided, defaults to None.
        lattice_vectors: Optional[np.ndarray]
            Lattice vectors of the snapshot, if applicable. If not provided, defaults to None.
        """
        self.name = label
        self.box_bounds = box_bounds
        self.atoms = atoms
        self.energy = energy

        if box_bounds is not None:
            # Orthorhombic box
            if self.box_bounds.shape[1] == 2:
                xlo, xhi = self.box_bounds[0]
                ylo, yhi = self.box_bounds[1]
                zlo, zhi = self.box_bounds[2]

                lx = xhi - xlo
                ly = yhi - ylo
                lz = zhi - zlo

                lattice_vectors = [
                    [lx, 0, 0],
                    [0, ly, 0],
                    [0, 0, lz]
                ]

            # Triclinic box
            elif self.box_bounds.shape[1] == 3:
                xlo_bound, xhi_bound, xy = self.box_bounds[0]
                ylo_bound, yhi_bound, xz = self.box_bounds[1]
                zlo_bound, zhi_bound, yz = self.box_bounds[2]

                xlo = xlo_bound - min(0.0, xy, xz, xy + xz)
                xhi = xhi_bound - max(0.0, xy, xz, xy + xz)
                ylo = ylo_bound - min(0.0, yz)
                yhi = yhi_bound - max(0.0, yz)
                zlo = zlo_bound
                zhi = zhi_bound

                lx = xhi - xlo
                ly = yhi - ylo
                lz = zhi - zlo

                lattice_vectors = [
                    [lx, 0, 0],
                    [xy, ly, 0],
                    [xz, yz, lz]
                ]

            self.lattice_vectors = lattice_vectors
        else:
            self.lattice_vectors = lattice_vectors

    @property
    def net_charge(self) -> float:
        """
        Calculate the net charge of the snapshot based on the charges of the atoms.

        Returns
        -------
        float
            The net charge of the snapshot.
        """
        net_charge = 0.0
        for atom in self.atoms:
            net_charge += atom.charge
        return net_charge

    # @property
    # def lattice_vectors(self)
    
    #     #Orthorhombic box
    #     if self.box_bounds.shape[1] == 2:
    
    #         xlo, xhi = self.box_bounds[0]
    #         ylo, yhi = self.box_bounds[1]
    #         zlo, zhi = self.box_bounds[2]
    
    #         lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
    
    #         lattice_vectors = [[lx, 0, 0],
    #                            [0, ly, 0],
    #                            [0, 0, lz]]
    
    #     # Triclinic box
    #     elif self.box_bounds.shape[1] == 3:
    
    #         xlo_bound, xhi_bound, xy = self.box_bounds[0]
    #         ylo_bound, yhi_bound, xz = self.box_bounds[1]
    #         zlo_bound, zhi_bound, yz = self.box_bounds[2]
    
    #         xlo = xlo_bound - min(0.0, xy, xz, xy+xz)
    #         xhi = xhi_bound - max(0.0, xy, xz, xy+xz)
    #         ylo = ylo_bound - min(0.0, yz)
    #         yhi = yhi_bound - max(0.0, yz)
    #         zlo = zlo_bound
    #         zhi = zhi_bound
    
    #         lx, ly, lz = xhi - xlo, yhi - ylo, zhi - zlo
    #         lattice_vectors = [[lx, 0, 0],
    #                            [xy, ly, 0],
    #                            [xz, yz, lz]]
    
    #     return lattice_vectors

    @property
    def symbols(self) -> List[str]:
        """Get the chemical symbols of the atoms in the snapshot."""
        symbols = [atom.element for atom in self.atoms]
        return symbols

    @property
    def needs_reference_calculation(self) -> bool:
        """Check if the snapshot needs a reference calculation."""
        if self.energy == 0:
            return True
        return False

    @property
    def element_count(self) -> Dict[str, int]:
        """Get the count of each element in the snapshot."""
        elements = set(self.symbols)
        counts_dict = {}
        for element in elements:
            counts_dict[element] = self.symbols.count(element)
        return counts_dict

    @property
    def positions(self) -> np.ndarray:
        """Get the positions of the atoms in the snapshot."""
        positions = [atom.position for atom in self.atoms]
        return np.array(positions)

    @property
    def direct(self) -> np.ndarray:
        """Get the direct lattice vectors."""
        return np.array(self.lattice_vectors)

    @property
    def inverse(self) -> np.ndarray:
        """Get the inverse lattice vectors."""
        return np.linalg.inv(self.direct)
    
    @property
    def volume(self) -> float:
        """Calculate the volume of the unit cell."""
        a, b, c = np.array(self.lattice_vectors)
        return a.dot(np.cross(b,c))

    def to_direct(self, coords) -> np.ndarray:
        """Convert Cartesian coordinates to fractional (direct) coordinates."""
        return np.dot(coords, self.direct)

    def to_fractional(self, coords) -> np.ndarray:
        """Convert Cartesian coordinates to fractional (direct) coordinates."""
        return np.dot(coords, self.inverse)

    def write(self, output, energy_force_unit='eV_A') -> None:
        """
        Writes snapshots in n2p2 format (input.data).
        Parameters
        ----------
        output: str
        The name of the output file.

        energy_force_unit: str
        Unit used for energy and forces. Choices are eV_A and Ha_Bohr
        Remember that the code assumes that source data units are in eV and A.
        """
        energy_prefactor = 1
        force_prefactor = 1
        if energy_force_unit == 'Ha_Bohr':
            energy_prefactor = eV_to_Ha
            force_prefactor = eV_A_to_Ha_B
        with open(output, 'a') as f:
            f.write('begin\n')
            f.write('comment {}\n'.format(self.name))
            if self.lattice_vectors is not None:
                for lv in self.lattice_vectors:
                    f.write('lattice  {:>14.10f}  {:>14.10f}  {:>14.10f}\n'.format(*lv))
            for atom in self.atoms:
                f.write('atom  {:>10.6f}  {:>10.6f}  {:>10.6f}  {:2s}  {:>+8.6f} '
                        ' {}  {:>10.6f}  {:>10.6f}  {:>10.6f}\n'.format(
                    atom.position[0], atom.position[1], atom.position[2],
                    atom.element, atom.charge, 0,
                    atom.force[0]*force_prefactor, atom.force[1]*force_prefactor, atom.force[2]*force_prefactor
                )
                )
            f.write('energy  {}\n'.format(self.energy * energy_prefactor))
            f.write('charge  {}\n'.format(self.net_charge))
            f.write('end\n')

    def write_xyz(self) -> None:
        """Writes the coordinates of atoms in a .xyz file"""
        fname = f'{self.name}.xyz'
        with open(fname, 'w') as f:
            f.write(f'{len(self.atoms)}\n')
            f.write(f'{self.name}\n')
            for atom in self.atoms:
                f.write('{:2s}  {:>10.6f}  {:>10.6f}  {:>10.6f}\n'.format(
                    atom.element,
                    atom.position[0], atom.position[1], atom.position[2],
                )
                )

    def write_res(self) -> None:
        """Writes the structure in res format (SG=1)"""
        self.write_xyz()
        xyz_to_res(
            xyz_fname=f'{self.name}.xyz',
            lattice_vectors=np.array(self.lattice_vectors)
        )

    @property
    def elements_group(self) -> OrderedDict:
        """ Group atoms by their element type."""
        elements_group_dict = OrderedDict()
        for atom in self.atoms:
            if atom.element not in elements_group_dict:
                elements_group_dict[atom.element] = {
                    'total_count': 0,
                    'positions': []
                }
            elements_group_dict[atom.element]['total_count'] += 1
            elements_group_dict[atom.element]['positions'].append(atom.position)
        return elements_group_dict

    def make_poscar(self, fractional=False, scaling="1.000000", comment="Made using n2p2 tools",
                    output_fname='POSCAR') -> None:
        poscar = []
        poscar.append(comment)
        poscar.append(scaling)  # Scaling factor for lattice vectors and atom coords (useful for e.g. cubic cells)

        for lv in self.lattice_vectors:
            poscar.append(f'{lv[0]:12.6f}  {lv[1]:12.6f}  {lv[2]:12.6f}')

        elements, counts = '', ''
        for element in self.elements_group.keys():
            elements += f'{element:>4s} '
            counts += f"{self.elements_group[element]['total_count']:4d} "
        poscar.append(elements)
        poscar.append(counts)

        ordered_coords = []
        if fractional:
            poscar.append("Direct")  # Fractional (direct) coords will follow
        else:
            poscar.append("Cartesian")  # Cartesian coords will follow

        for element in self.elements_group.keys():
            for position in self.elements_group[element]['positions']:
                poscar.append(f'{position[0]:12.6f}  {position[1]:12.6f}  {position[2]:12.6f}')

        with open(output_fname, 'w') as f:
            for line in poscar:
                f.write(f'{line}\n')

    @property
    def reciprocal_lattice_vectors(self) -> List[np.ndarray]:
        a1, a2, a3 = np.array(self.lattice_vectors)
        v = np.dot(a1, np.cross(a2, a3))
        b1 = np.cross(a2, a3) / v
        b2 = np.cross(a3, a1) / v
        b3 = np.cross(a1, a2) / v
        return b1, b2, b3

    def make_kpoints(self, numk=None, kspacing=None, output_fname = 'KPOINTS') -> None:
        '''
        Generated KPOINTS file based on kspacing or number of kpoints in each direction.
        Either kspacing or numk should be specified.
        kspacing = 0.05 is an optimized choice for molecular systems.
        If numk is a single number (int) the same number of kpoints for all directions.
        In numk is a list of three numbers (int), those numbers will be used to sample
        reciprocal space in b1, b2, and b3 direction.
        '''

        self.number_of_kpoints = 1

        kpoints = []
        if numk and kspacing:
            raise Exception("Must specify ONE of A) number of k-points (a single int or 3 ints), or B) a k-spacing")

        kpoints.append(f'K-point grid for {self.name}')
        kpoints.append("0")  # Number of kpoints; 0 indicates automatic generation
        # If a kspacing is defined, work out int number of kpoints in each direction
        if kspacing is not None:
            kpoints.append("Gamma")
            temp_line = ''
            for kv in self.reciprocal_lattice_vectors:
                norm_ = np.linalg.norm(kv)
                # 1- Minimum K-point in each direction is 1
                # 2- Want to round up as denser k-grid is desired
                nk = int(np.ceil(max(1, norm_/kspacing+0.5)))
                self.number_of_kpoints *= nk
                temp_line += f'{nk:4d} '
            kpoints.append(temp_line)

        elif isinstance(numk, int):
            kpoints.append("Auto")
            kpoints.append(str(numk))
        elif isinstance(numk, list) and len(numk) == 3:
            kpoints.append("Gamma")
            temp_line = ''
            for num in numk:
                temp_line += f'{num:4d}'
            kpoints.append(temp_line)
            # kpoints.append(["  ".join(str(k) for k in numk)])
            # print "0. 0. 0."  # shift of the k-point grid; optional line
        with open(output_fname, 'w') as f:
            for line in kpoints:
                f.write(f'{line}\n')

    def make_potcar(self, output_fname='POTCAR') -> None:
        '''
        concatenated POTCARs of elements in the unit cell.
        '''
        with open(output_fname, 'w+') as f:
            for element in self.elements_group.keys():
                with open(os.path.join(potcar_path, str(element), 'POTCAR'), 'r') as element_potcar:
                    for line in element_potcar:
                        f.write(line)

    def make_incar_sp(self, output_fname='INCAR', ncpu=1) -> None:
        '''
        To have self.number_of_kpoints assigned, this function must be run after make_kpoints
        '''
        if not self.number_of_kpoints:
            raise Exception("self.number_of_kpoints not set yet. Run make_kpoints() first!")

        kpar_ = int(gcd(int(ncpu/4), self.number_of_kpoints))
        natoms = len(self.atoms)
        with open(output_fname, 'w') as f:
            f.write('PREC = Accurate      #  precision normal\n')
            f.write('ENCUT = 500          #  plane-wave energy cut-off (basis size)\n')
            f.write('LREAL = Auto         #  real space projection yes / no\n')
            f.write('ISIF = 0             #  relaxation of: atoms, don\'t calculate stress tensor\n')
            f.write('IVDW = 0             #  dispersion correction (12 = GD3BJ, 1 = D2)\n')
            f.write('NELMIN = 5           #  do a minimum of four electronic steps\n')
            f.write(f'EDIFF  = {natoms*1e-7:1.0E}    #  convergence of total energy (not per atom) in SCF\n')
            f.write('EDIFFG = -0.03       #  convergence of energy (forces if EDIFFG -ve) in geom opt\n')
            f.write('NSW    = 0           #  number of geom opt steps\n')
            f.write('IBRION =  2          #  use CG algorithm\n')
            f.write(f'KPAR = {kpar_}        #  parallelise over k-points\n')
            f.write('NCORE = 4             #  NPAR orbitals per node\n')
            f.write('LWAVE = .FALSE.      #  don\'t write out wavefunction\n')
            f.write('LCHARG = .FALSE.     #  don\'t write out charge density\n')

    def make_jobfile(self, ncpu, whichHPC='young', output_fname='job.sub') -> None:
        self.jobfile = os.path.realpath(output_fname)
        self.vasp_folder = os.path.dirname(self.jobfile)
        if whichHPC == 'young':
            with open(output_fname, 'w') as f:
                f.write('#!/bin/bash -l\n')
                f.write('#$ -P Gold\n')
                f.write('#$ -A Soton_allocation\n')
                f.write('#$ -l h_rt=00:30:00\n')
            #    f.write('#$ -l mem=2G\n')
                f.write(f"#$ -N {self.name}\n")
                f.write(f'#$ -pe mpi {ncpu}\n')
                f.write('##$ -m be\n')
                f.write(f'#$ -wd {self.vasp_folder}\n')
                f.write(f'cd {self.vasp_folder}\n')
                f.write('module purge\n')
                f.write('module load --silent default-modules\n')
                f.write('module load vasp/5.4.4-18apr2017/intel-2017-update1\n')
                f.write('gerun vasp_std > vasp_output.$JOB_ID\n')
    def submitjob(self, whichHPC='young') -> None:

        if not self.jobfile:
            raise Exception('run make_jobfile() first!')
        import subprocess
        if whichHPC == 'young':
            submission_command=f'qsub {self.jobfile}'
            result = subprocess.check_output(submission_command, shell=True)
            self.job_id = int(result.split()[2])


    def generate_vasp_inputs(self, output_folder='.', kspacing=0.05, ncpu=1) -> None:
        os.makedirs(output_folder, exist_ok=True)
        self.make_poscar(
            output_fname=f'{output_folder}/POSCAR',
            comment=f'{self.name}'
        )
        self.make_potcar(output_fname=f'{output_folder}/POTCAR')
        self.make_kpoints(output_fname=f'{output_folder}/KPOINTS', kspacing=kspacing)
        self.make_incar_sp(output_fname=f'{output_folder}/INCAR', ncpu=ncpu)
        self.make_jobfile(output_fname=f'{output_folder}/job.sub', ncpu=ncpu)

    def run_vasp(self, vasp_folder, ncpu) -> None:
        if call("which vasp_std", shell=True): #if vasp_std can't be found
            raise Exception('Load VASP and dependencies first')
        self.vasp_folder = os.path.realpath(vasp_folder)
        call(f"mpiexec -n {ncpu} vasp_std > log", cwd=self.vasp_folder, shell=True)
