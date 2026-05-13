from cspy.chem.multipole import DistributedMultipoles
import numpy as np
import logging


def hexadecapole_term(h, x, y, z, xx, xy, xz, yy, yz, zz, x3, y3, z3, x4, y4, z4, r5):
    v = h[0] * 0.125 * (35 * z4 - 30 * zz + 3)
    v += np.sqrt(5 / 8) * (h[1] * (7 * x * z3 - 3 * xz) + h[2] * (7 * y * z3 - 3 * yz))
    v += h[3] * np.sqrt(5 / 16) * (xx - yy) * (7 * zz - 1)
    v += h[4] * np.sqrt(5 / 4) * xy * (7 * zz - 1)
    v += z * np.sqrt(35 / 8) * (h[5] * (x3 - 3 * x * yy) + h[6] * (3 * xx * y - y3))
    v += h[7] * np.sqrt(35 / 64) * (x4 - 6 * xx * yy + y4)
    v += h[8] * np.sqrt(35 / 4) * (x3 * y - x * y3)
    return v / r5


def charge_term(q, r):
    return q / r


def dipole_term(d, xyz, r2):
    v = np.dot(d[[1, 2, 0]], xyz.T) / r2
    return v


def quadrupole_term(q, xx, yy, zz, xy, xz, yz, r2, r3):
    sqrt3 = np.sqrt(3)
    v = q[0] * 0.5 * (3 * zz - 1)
    v += sqrt3 * (q[1] * xz + q[2] * yz + q[4] * xy)
    v += np.sqrt(0.75) * q[3] * (xx - yy)
    v /= r3
    return v


def octapole_term(o, x, y, z, xx, yy, zz, xxx, yyy, zzz, r4):
    # octapole
    v = o[0] * 0.5 * (5 * zzz - 3 * z)
    v += np.sqrt(3 / 8) * (o[1] * x + o[2] * y) * (5 * zz - 1)
    v += o[3] * np.sqrt(15 / 4) * z * (xx - yy)
    v += o[4] * np.sqrt(15) * x * y * z
    v += o[5] * np.sqrt(5 / 8) * (xxx - 3 * x * yy)
    v += o[6] * np.sqrt(5 / 8) * (3 * xx * y - yyy)
    v /= r4
    return v


def compare_error(esp, ref="charges.out"):
    ref = np.loadtxt(ref)
    diff = esp - ref[:, 3]
    print("MAE: ", np.mean(np.abs(diff)))


quad_vals = np.array(
    [
        [1.740905, 1.698470],
        [1.551669, 1.412429],
        [1.290123, 1.471799],
        [2.650102, 2.290647],
        [2.144712, 1.970086],
        [2.207267, 2.374609],
    ]
)


def electrostatic_potential(multipoles, grid):
    mol = multipoles.molecules[0]
    positions = np.array(mol.positions)
    BOHR = 1.88973
    positions *= BOHR
    nuc = [x.atomic_number for x in mol.elements]
    charges = np.array([x.charge for x in mol.multipoles])
    dipoles = np.array([x.dipole for x in mol.multipoles])
    quadrupoles = np.array([x.quadrupole for x in mol.multipoles])
    octapoles = np.array([x.octapole for x in mol.multipoles])
    hexadecapoles = np.array([x.hexadecapole for x in mol.multipoles])
    esp = np.zeros(grid.shape[0])
    for pos, q1, d, qq, o, h in zip(
        positions, charges, dipoles, quadrupoles, octapoles, hexadecapoles
    ):
        xyz = grid - pos
        r = np.linalg.norm(xyz, axis=1)
        xyz /= r[:, np.newaxis]
        x, y, z = xyz.T
        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        yz = y * z
        xz = x * z
        x3 = xx * x
        y3 = yy * y
        z3 = zz * z
        r2 = r * r
        r3 = r2 * r
        r4 = r3 * r
        esp += charge_term(q1, r)
        esp += dipole_term(d, xyz, r2)
        esp += quadrupole_term(qq, xx, yy, zz, xy, xz, yz, r2, r3)
        esp += octapole_term(o, x, y, z, xx, yy, zz, x3, y3, z3, r4)
        x4 = x3 * x
        y4 = y3 * y
        z4 = z3 * z
        r5 = r4 * r
        esp += hexadecapole_term(
            h, x, y, z, xx, xy, xz, yy, yz, zz, x3, y3, z3, x4, y4, z4, r5
        )
    return esp * 27.211386245


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mults", type=str, help="Multipole file")
    parser.add_argument("vpot", type=str, help="Grid file")
    parser.add_argument("ref", type=str, help="reference")
    args = parser.parse_args()

    logging.basicConfig(level="DEBUG")
    multipoles = DistributedMultipoles.from_dma_file(args.mults)
    grid = np.loadtxt(args.vpot, skiprows=1)
    from cspy.util.timer import Timer

    timing = Timer()
    with timing:
        esp = electrostatic_potential(multipoles, grid)
    print("Took:", timing.elapsed)
    compare_error(esp, ref=args.ref)
    np.savetxt("out.vpot", np.c_[grid, esp])


if __name__ == "__main__":
    main()
