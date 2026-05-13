from cspy.crystal import Crystal
import logging
import os
import sys
import argparse
import numpy as np
from math import gcd

LOG = logging.getLogger(__name__)

potcar_path="/home/rcdb1c20/VASP_POTENTIALS" # needs updating
kspacing=0.05

def _kpoints_per_klength(klength,kspacing):
    # Must have at least 1 k-point
    nk_raw = max(1,klength/kspacing+0.5)
    # Want to round up as kspacing is maximum spacing desired
    return int(np.ceil(nk_raw))

def make_poscar(crystal,fractional=True,scaling="1.000000",comment="Made using CSPy"):
    poscar = []
    latt_vecs = crystal.unit_cell.lattice
    poscar.append(comment)
    poscar.append(scaling)   # Scaling factor for lattice vectors and atom coords (useful for e.g. cubic cells)
    for i in range(0,3):
        poscar.append("  "+"{:14.8f} {:14.8f} {:14.8f}".format(*latt_vecs[i]))
    poscar.append("  "+("{:>4s}"*len(elem_set)).format(*[str(el) for el in elem_set])) # Can explicitly list elements in POSCAR, above element counts
    poscar.append("  "+("{:4d}"*len(elem_set)).format(*elem_counts))

    uc_atoms = crystal.unit_cell_atoms()
    ordered_uc_atoms = []
    for el in elem_set:
        ordered_uc_atoms += [i for i,e in zip(uc_atoms["asym_atom"],uc_atoms["element"]) if e == el.atomic_number]

    ordered_coords = []
    if fractional:
        poscar.append("Direct")  # Fractional (direct) coords will follow
        for i in ordered_uc_atoms: 
            ordered_coords.append(uc_atoms["frac_pos"][i])
    else:
        poscar.append("Cartesian")  # Cartesian coords will follow 
        for i in ordered_uc_atoms: 
            ordered_coords.append(uc_atoms["cart_pos"][i])
    LOG.debug("Ordered coordinates in make_poscar are:")
    LOG.debug("\n".join([str(c) for c in ordered_coords]))
    for coord in ordered_coords:
        poscar.append("  "+"{:14.8f} {:14.8f} {:14.8f}".format(*coord)) # unpack array as tuple to format
    return poscar

def make_kpoints(crystal,numk=None,kspacing=None):
    kpoints = []
    if numk and kspacing:
        raise Exception("Must specify ONE of A) number of k-points (a single int or 3 ints), or B) a k-spacing")

    kpoints.append("K-point grid for " + crystal.titl) #args.filename.split('/')[-1]) - args not defined?

    kpoints.append("0") # Number of kpoints; 0 indicates automatic generation
    # If a kspacing is defined, work out int number of kpoints in each direction
    if kspacing is not None:
        kpoints.append("Gamma")
        k = [0., 0., 0.]
        k_vec_norms = [np.linalg.norm(kvec) for kvec in crystal.unit_cell.reciprocal_lattice]
        nkvec = [_kpoints_per_klength(knorm,kspacing) for knorm in k_vec_norms]
        kpoints.append("{:4d} {:4d} {:4d}".format(*nkvec) )
    elif isinstance(numk,int):
        kpoints += ["Auto"]
        kpoints += [str(numk)]
    elif isinstance(numk,list) and len(numk)==3:
        kpoints += ["Gamma"]
        kpoints += ["  ".join(str(k) for k in numk)]
        #print "0. 0. 0."  # shift of the k-point grid; optional line
    return kpoints

def write_potcar(elements):
    with open("POTCAR","w+") as final_potcar:
        for elem in elements:
            with open(os.path.join(potcar_path,str(elem),"POTCAR"),"r") as elem_potcar:
                for line in elem_potcar:
                    final_potcar.write(line)
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename",
                        type=str,
                        help="""A .res (SHELX) or CIF file containing the P1 crystal structure 
                        to prepare VASP inputs for (filetype determined by extension)."""
            )
    parser.add_argument("--kspacing",
                        type=float,
                        help="A float value indicating the maximum k-point spacing in reciprocal space.",
                        default=0.05
            )
    parser.add_argument("--ncpu", 
            type=int,
            help="The number of total cores that will be used for VASP job.",
            default=40
            )
    parser.add_argument("--jobfile",
            type=str,
            help="Use it to specify the type of the job submission script of the HPC",
            default='young'
            )
    parser.add_argument("--submitjob",
            help="Use it to submit the job after generating the job file",
            action='store_true'
            )
    parser.add_argument("--optimisation-steps",
            type=str,
            help="Use it to submit the job after generating the job file",
            default='0'
            )
    args = parser.parse_args()
    crys = Crystal.load(args.filename)
    try:
        crys = crys.standardized()
    except:
        LOG.info('Niggli cell reduction failed - continuing with original cell')
    crys = crys.as_P1()

    # Element (atom) lists, set of elements, and count of each
    # are all needed inside functions also; define them here
    elem_list = crys.asymmetric_unit.elements
    elem_set = set(elem_list)
    elem_counts = [elem_list.count(el) for el in elem_set]

    with open("POSCAR","w+") as f:
        f.write("\n".join(make_poscar(crys,comment=args.filename.split('/')[-1] + " made using CSPy")))

    with open("KPOINTS","w+") as f:
        f.write("\n".join(make_kpoints(crys,kspacing=args.kspacing)))
    write_potcar(elem_set)

    num_kpoints = 1
    with open('KPOINTS', 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        parts = last_line.split()
        for part in parts:
            num_kpoints *= int(part)

    ncore = 8
    kpar_ = int(gcd(int(args.ncpu/ncore), num_kpoints))

    natoms = crys.asymmetric_unit.atomic_numbers.shape[0]

    with open('INCAR', 'w') as f:
        f.write('ISTART = 0           #  \n')
        f.write('ICHARG = 0           #  \n')
        f.write('PREC = Accurate      #  precision normal\n')
        f.write('ENCUT = 500          #  plane-wave energy cut-off (basis size)\n')
        f.write('LREAL = Auto         #  real space projection yes / no\n')
        f.write('ISIF = 0             #  relaxation of: atoms, don\'t calculate stress tensor\n')
        f.write('ISYM = 0             \n')
        f.write('#SYMPREC = 1E-4       \n')
        f.write('IVDW = 12            #  dispersion correction (12 = GD3BJ, 1 = D2)\n')
        f.write('GGA = PE             #  PBE XC functional\n')
        f.write('NELM = 300           #  do a minimum of four electronic steps\n')
        f.write(f'EDIFF  = {natoms*1e-7:1.0E}    #  convergence of total energy (not per atom) in SCF\n')
        f.write('EDIFFG = -0.03       #  convergence of energy (forces if EDIFFG -ve) in geom opt\n')
        f.write("NSW    = "+str(args.optimisation_steps)+"         #  number of geom opt steps\n")
        f.write('IBRION = -1          #  use QN algorithm\n')
        f.write('POTIM = 0.3         #  QN step\n')
        f.write(f'KPAR = {kpar_}        #  parallelise over k-points\n')
        f.write(f'NCORE = {ncore}\n')
        f.write('LWAVE = .FALSE.      #  don\'t write out wavefunction\n')
        f.write('LCHARG = .FALSE.     #  don\'t write out charge density\n')

    LOG.info("Finished writing INCAR, POSCAR, KPOINTS, and POTCAR for "+args.filename)

    #if args.jobfile == 'young':
    #    import os
    #
    #    with open('job.sub', 'w') as f:
    #        f.write('#!/bin/bash -l\n')
    #        f.write('#$ -P Gold\n')
    #        f.write('#$ -A Soton_allocation\n')
    #        f.write('#$ -l h_rt=00:30:00\n')
    #    #    f.write('#$ -l mem=2G\n')
    #        f.write(f"#$ -N {args.filename.split('/')[-1][:-4]}\n")
    #        f.write(f'#$ -pe mpi {args.ncpu}\n')
    #        f.write('##$ -m be\n')
    #        f.write(f'#$ -wd {os.getcwd()}\n\n')
    ##        f.write(f'OMP_NUM_THREADS={int(args.ncpu / kpar_)}')
    #        f.write('cd $SGE_O_WORKDIR\n')
    #        f.write('module purge\n')
    #        f.write('module load --silent default-modules\n')
    #        f.write('module load vasp/5.4.4-18apr2017/intel-2017-update1\n')
    #        f.write('gerun vasp_std > vasp_output.$JOB_ID\n')
    #    LOG.info("Finished writing job.sub for "+args.filename)
    #
    #    if args.submitjob:
    #        os.system(f'qsub job.sub')
    #        LOG.info(f'Job submitted for {args.filename}')

if __name__ == "__main__":
    main()