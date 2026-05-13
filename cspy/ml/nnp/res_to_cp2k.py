from cspy import Crystal
import numpy as np


def res_to_cp2k(input_fname, output_fname=None, supercell=[1,1,1]):
    supercell = np.array(supercell)
    c = Crystal.load(input_fname)
    c = c.as_P1()

    if not output_fname:
        output_fname = input_fname[:-4] + '.cp2k'

    with open(output_fname, 'w') as f:
        f.write('&CELL\n')
        lvs = c.unit_cell.lattice
        supercell_lvs = lvs * supercell[:, np.newaxis]

        f.write(f'  A {supercell_lvs[0][0]:12.8f} {supercell_lvs[0][1]:12.8f} {supercell_lvs[0][2]:12.8f}\n')
        f.write(f'  B {supercell_lvs[1][0]:12.8f} {supercell_lvs[1][1]:12.8f} {supercell_lvs[1][2]:12.8f}\n')
        f.write(f'  C {supercell_lvs[2][0]:12.8f} {supercell_lvs[2][1]:12.8f} {supercell_lvs[2][2]:12.8f}\n')
        f.write('  PERIODIC XYZ\n')
        f.write('&END CELL\n')
        f.write('&COORD\n')
        na, nb, nc = supercell
        for i in range(na):
            for j in range(nb):
                for k in range(nc):
                    for e, p in zip(c.asymmetric_unit.elements, c.to_cartesian(c.asymmetric_unit.positions)):
                        p += i*lvs[0] + j*lvs[1] + k*lvs[2]
                        f.write(f'  {e}   {p[0]:12.8f} {p[1]:12.8f} {p[2]:12.8f}\n')
        f.write('&END COORD')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
            "files",
            type=str,
            nargs='+',
            help='res files to convert',
    )

    parser.add_argument(
            '--na',
            type=int,
            metavar='',
            help='For supercell generation purposes: repetition in lattice vector a direction',
            default = 1
    )
    
    parser.add_argument(
            '--nb',
            type=int,
            metavar='',
            help='For supercell generation purposes: repetition in lattice vector b direction',
            default = 1
    )

    parser.add_argument(
            '--nc',
            type=int,
            metavar='',
            help='For supercell generation purposes: repetition in lattice vector c direction',
            default = 1
    )


    parser.add_argument(
            '--output',
            metavar='',
            help='Output filename',
            default=None
    )


    args = parser.parse_args()
    
    for file in args.files:
        res_to_cp2k(file, args.output, supercell = [args.na, args.nb, args.nc])
