import re
import math as mt

import numpy as np

def map_point_to_sphere(numpyarray):
    bp = []
    factor = 1.0
    for l in numpyarray:
        bp += [factor*mt.cos(l)]
        factor *= mt.sin(l)
    bp = bp+[factor]
    bp.reverse()
    return np.array(bp)

class Point:
    def __init__(self, coords, label=0, reference=None):
        self.coords = coords
        self.label = label
        self.n = len(coords)
        self.reference = reference

    def __repr__(self):
        return str(self.coords) + ", " + str(self.label)

class Angles():
    """An object to store angular data"""

    def __init__(self, units="degrees"):

        self.dataset = None
        self.units = units
        self.file = None

    def __repr__(self):
        return str(self.file)+", "+str(self.units)

    def add(self, b, units="degrees", data_id=''):
        """
        :param b: numpy array representing dataset
        :param units:
        :return: adds dataset to an object
        """
        if self.dataset is None:
            self.units = units
            self.file = data_id
            if type(b).__module__ == np.__name__:
                self.dataset = b
        else:
            print("your data is not a numpy object")

    def read_csv(self, filename, delimiter=',', units="degrees"):
        """reading datasetfrom csv format
        :param filename: csv file containing output results
        :param: delimiter: by default is ',' for comma separated values
        :return: dataset attribute will contain datafromfrom file specified by filename
        """
        if self.dataset is None:
            self.file = filename
            self.dataset = np.loadtxt(filename, delimiter=delimiter)
            self.units = units
        else:
            print("The dataset of this object is not empty")

    def clean(self):
        """making an object empty"""
        self.dataset = None
        self.file = None
        self.units = "degrees"

    def read_amber(self, filename):
        """
        readingfrom traj output
        :param filename: file containing output results
        :return: dataset attribute will contain datafromfrom file specified by filename
        """
        if self.dataset is None:
            try:
                f = open(filename)
                data = []
                for line in f:
                    x = np.array([float(i) for i in (line.strip()).split()])
                    data.append(x)
                self.dataset = np.array(data)[:, 1:]
                self.units = "degrees"
                self.file = filename
            except IOError:
                print("there is no such file")
            except ValueError:
                print("something wrong with data")
        else:
            print("The dataset of this object is not empty")

    def read_charmm(self, filename):
        """reading dataset for charmm output for torsions (similar to Gromacs)
        :param filename: file containing output results
        :return: dataset attribute will contain datafromfrom file specified by filename
        """
        if self.dataset is None:
            try:
                self.units = "degrees"
                self.file = filename
                f = open(filename)
                data = []
                for line in f:
                    if "#" not in line and line.strip() != "" and "@" not in line:
                        x = np.array([float(i) for i in (line.strip()).split()])
                        data.append(x)
                f.close()
                data = np.array(data)
                self.dataset = np.array(data[:, 1:])
            except IOError:
                print("there is no such file")
            except ValueError:
                print("something wrong with data")
        else:
            print("The dataset of this object is not empty")

    def add_columns(self, b):
        """ adding columnsfrom dataset of object bfrom the right side column wise, with row names as in current object.
        :param b: an Angles object   """
        if isinstance(b, Angles):
            if b.dataset.shape[0] == self.dataset.shape[0]:
                if b.units == self.units:
                    self.dataset = np.concatenate((self.dataset, b.dataset), axis=1)
                    self.file = self.file + " " + b.file
                else:
                    print("units of two datasets are different")
            else:
                print("Number of rows in both datasets are different!")
        else:
            print("object containing dataset to add is not instance of class Angles")

    def select(self, *args):
        """
        :param b: an Angle object
        :param args: column numbers to select for a new dataset (separated by comma)
        :return: a new Angle object containing selected columns
        """
        angle_new = Angles(self.units)
        numpy_object = self.dataset[:, args[0]-1]
        for i in args[1:]:
            numpy_object = (np.vstack((numpy_object, self.dataset[:, i-1]))).T
        angle_new.add(numpy_object, units=self.units, data_id=self.file)
        return angle_new

    def add_rows(self, b):
        """ adding columnsfrom dataset of object bfrom the right side column wise, with row names as in current object.
        :param b: an Angles object   """
        if isinstance(b, Angles):
            if b.dataset.shape[1] == self.dataset.shape[1]:
                if self.units == b.units:
                    self.dataset = np.concatenate((self.dataset, b.dataset), axis=0)
                    self.file = self.file + " " + b.file
                else:
                    print("units of two datasets are different")
            else:
                print("Number of columns in both datasets are different!")
        else:
            print("object containing dataset to add is not instance of class Angles")

    def to_degrees(self):
        """
        :return:dataset converted to degrees
        """
        if self.units == "radians":
            self.dataset = np.degrees(self.dataset)
            self.units = "degrees"

    def to_radians(self):
        """
        :return: dataset converted to radians
        """
        if self.units == "degrees":
            self.dataset = np.radians(self.dataset)
            self.units = "radians"

    def unitsphere(self):
        """:return: mapping on to the sphere
        """
        if self.units == "degrees":
            return list(map(map_point_to_sphere, np.radians(self.dataset)))
        else:
            return np.array(list(map(map_point_to_sphere, self.dataset)))


def vector_prod(a, b):
    """ a helper function for obtaining vector product of two vectors
    :param a: one dimensional numpy array (in 3D space)
    :param b: one dimensional numpy array (in 3D space)
    :return: one dimensional numpy array (in 3D space)
    """
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return np.array([x, y, z])


def find_atom_coord(atom_name, residue_number, line):
    """ searching atomic coordinates in a given line
    :param atom_name: atom name
    :param residue_number: residue number
    :param line: given text linefrom pdb
    :return: numpy arrya containing coordinates of the atom if found, or None
    """
    found = re.search("ATOM\s+\d+\s+" + atom_name + "\s+\w+\s+\w+\s+" + residue_number, line)
    if found:
        coord_found = re.search("\d+\.\d*\s+\d+\.\d*\s+\d+\.\d*", line)
        coord = np.array([float(i) for i in (coord_found.group(0).split(' ')) if i.strip() != ""])
        return coord
    else:
        found = re.search("HETATM\s+\d+\s+" + atom_name + "\s+\w+\s+\w+\s+" + residue_number, line)
        if found:
            coord_found = re.search("\d+\.\d*\s+\d+\.\d*\s+\d+\.\d*", line)
            coord = np.array([float(i) for i in (coord_found.group(0).split(' ')) if i.strip() != ""])
            return coord
    return None


def get_dihedral(pdbname, atom1, atom2, atom3, atom4):
    """calculating dihedral angle for given atoms in PDB"""
    f1, f2, f3, f4 = None, None, None, None
    r1 = re.search("\d+@", atom1)
    if r1:
        r1 = r1.group(0)[:-1]

    r2 = re.search("\d+@", atom2)
    if r2:
        r2 = r2.group(0)[:-1]

    r3 = re.search("\d+@", atom3)
    if r3:
        r3 = r3.group(0)[:-1]

    r4 = re.search("\d+@", atom4)
    if r4:
        r4 = r4.group(0)[:-1]

    name1 = re.search("@\w+", atom1)
    if name1:
        name1 = name1.group(0)[1:]

    name2 = re.search("@\w+", atom2)
    if name2:
        name2 = name2.group(0)[1:]

    name3 = re.search("@\w+", atom3)
    if name3:
        name3 = name3.group(0)[1:]

    name4 = re.search("@\w+", atom4)
    if name4:
        name4 = name4.group(0)[1:]

    if r1 and r2 and r3 and r4 and name1 and name2 and name3 and name4:
        f = open(pdbname, "r")
        for line in f:
            if f1 is None:
                f1 = find_atom_coord(name1, r1, line)
            if f2 is None:
                f2 = find_atom_coord(name2, r2, line)
            if f3 is None:
                f3 = find_atom_coord(name3, r3, line)
            if f4 is None:
                f4 = find_atom_coord(name4, r4, line)
        f.close()
    else:
        print("Warning: Something wrong with atom names")
        return None
    if f1 is not None and f2 is not None and f3 is not None and f4 is not None:
        if len(f1) == 3 and len(f2) == 3 and len(f3) == 3 and len(f4) == 3:
            one = f2 - f1
            two = f3 - f2
            three = f4 - f3
            n1 = vector_prod(one, two)
            n2 = vector_prod(two, three)
            cos = np.inner(n1, n2) / np.sqrt(np.inner(n1, n1) * np.inner(n1, n1))
            if cos < -1:
                return 180.0
            elif cos > 1:
                return 0.0
            else:
                return mt.acos(cos)*180.0/np.pi
        else:
            print("Something wrong with PDB or this function, please check PDB and/or report the bug")
    else:
        print("Warning: One of atoms specified is non existing")
    return None


