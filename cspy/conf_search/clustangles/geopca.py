""" Module for performing principal geodesic analysis """

import random as random
from .inputs import *


def chi(x, v, point):
    """ a helper function in GeoPCA calculation """
    a = mt.sqrt(np.power((np.inner(x, point)), 2) + np.power((np.inner(v, point)), 2))
    if a > 1:
        return 1.
    return a


def epsilon(chis):
    """ a helper function in GeoPCA calculation """
    if chis == 1.:
        return 1.
    if chis != 1.:
        if chis > 1.:
            print("warning")
        return mt.acos(chis) / (chis * mt.sqrt(1 - np.power(chis, 2)))


def lambda1(x, v, pointset):
    """ first Lagrange mulriplier """
    suma = 0
    for point in pointset:
        c = epsilon(chi(x, v, point))
        suma = suma + c * np.power(np.dot(x, point), 2)
    return suma


def lambda2(x, v, pointset):
    """ second Lagrange multiplier """
    suma = 0
    for point in pointset:
        c = epsilon(chi(x, v, point))
        suma = suma + c * np.dot(x, point) * np.dot(v, point)
    return suma


def lambda3(x, v, pointset):
    """ third Lagrange multiplier """
    suma = 0
    for point in pointset:
        c = epsilon(chi(x, v, point))
        suma = suma + c * np.power(np.dot(v, point), 2)
    return suma


def psi1(lambda1, lambda2, lambda3, x, v, pointset):
    """ a helper function in GeoPCA calculation """
    suma = np.array([0])
    for point in pointset:
        c = epsilon(chi(x, v, point))
        suma = suma + np.array([point]) * c * (lambda3 * np.inner(x, point) - lambda2 * np.inner(v, point)) / (
        lambda1 * lambda3 - np.power(lambda2, 2))
    return suma


def psi2(lambda1, lambda2, lambda3, x, v, pointset):
    """ a helper function in GeoPCA calculation """
    suma = np.array([0])
    for point in pointset:
        c = epsilon(chi(x, v, point))
        suma = suma + np.array([point]) * c * (lambda2 * np.dot(x, point) - lambda1 * np.dot(v, point)) / (
        -lambda1 * lambda3 + np.power(lambda2, 2))
    return suma


def fu(x, v, dataset):
    """function to optimize"""
    suma = 0
    for point in dataset:
        suma = suma + mt.acos(
            mt.sqrt(np.inner(x, point) * np.inner(x, point) + np.inner(v, point) * np.inner(v, point)))
    return suma


def ab(t, i, x, v):
    """ a helper function in GeoPCA calculation """
    return np.inner(x, i) * mt.cos(t) + np.inner(v, i) * mt.sin(t)


def bb(t, i, x, v):
    """ a helper function in GeoPCA calculation """
    return np.inner(v, i) * mt.cos(t) - np.inner(x, i) * mt.sin(t)


def chi2(t, w, i, x, v):
    """ a helper function in GeoPCA calculation """
    a = mt.sqrt(np.power(ab(t, i, x, v), 2) + np.power((np.inner(w, i)), 2))
    if a > 1:
        return 1
    return a


def phi(w, t, dataset, x, v):
    """ a helper function in GeoPCA calculation """
    suma = np.array([0])
    sum = 0
    for point in dataset:
        c = epsilon(chi2(t, w, point, x, v))
        suma = suma + c * np.inner(w, point) * point
        sum = sum + c * np.power(np.inner(w, point), 2)
    return suma / sum


def projection_1(dataset, x, v):
    """ projection of the data on the given geodesic """
    proj = []
    for point in dataset:
        proj = proj + [(np.inner(x, point) * x + np.inner(v, point) * v) / mt.sqrt(
            np.power(np.inner(x, point), 2) + np.power(np.inner(v, point), 2))]
    return proj


def variance(dataset, y, w):
    """variance of the data with respect to the given geodesics """
    suma = 0
    for point in dataset:
        suma = suma + (np.inner(y, point) * np.inner(y, point) + np.inner(w, point) * np.inner(w, point)) / len(dataset)
    return suma[0]


def get_angles(dataset, v, x, t, origin):
    """ coordinates of the datapoints in new angles given by 
    the first two principal geodesics """
    angles = []
    for point in dataset:
        if mt.asin(np.inner(point, v) / np.inner(origin, origin)) > 0:
            angles = angles + [mt.acos(np.inner(point, x) / np.inner(origin, origin)) - t]
        else:
            angles = angles + [-mt.acos(np.inner(point, x) / np.inner(origin, origin)) - t]
    return np.array(angles)


def dist(x, v):
    """ a helper function in GeoPCA calculation """
    return [mt.sqrt(np.inner(x, x)), mt.sqrt(np.inner(v, v))]


def dist1(w, t):
    """ a helper function in GeoPCA calculation """
    return [mt.sqrt(np.inner(w, w)), abs(t)]


def indi(name):
    """ a helper function in GeoPCA calculation """
    z = np.zeros(len(name[0]))
    num = np.copy(z)
    while np.array_equal(num, z):
        nume = random.randint(0, (len(name)) - 1)
        num = name[nume]

    return num


def indu(name, indigo):
    """ a helper function in GeoPCA calculation """
    for i in range(len(indigo)):
        if (indigo[i]) != 0:
            ind = i
            break
    return ind


def initializeFirst(name, indigo, ind):
    """initialization of the first geodesic"""

    c = sum(indigo)
    v1 = [1.0 for i in indigo]
    c = (indigo[ind] - c) / indigo[ind]
    v1[ind] = c
    norm = mt.sqrt(np.inner(v1, v1))
    v1 = np.array(v1) / norm

    return indigo, v1


def runFirst(d1, d2, name, x, v, first_stop):
    """ search of the first principal geodesics """
    while d1 > first_stop and d2 > first_stop:
        x1 = x
        v1 = v
        l1 = lambda1(x, v, name)
        l2 = lambda2(x, v, name)
        l3 = lambda3(x, v, name)

        c0 = psi1(l1, l2, l3, x, v, name)
        c0 = c0 / mt.sqrt(np.inner(c0, c0))
        v = psi2(l1, l2, l3, x, v, name) - np.inner(psi2(l1, l2, l3, x, v, name), c0) * c0
        v = v / mt.sqrt(np.inner(v, v))
        x = c0

        d1 = dist(x1 - x, v1 - v)[0]
        d2 = dist(x1 - x, v1 - v)[1]

    return x, v


def initializeSecond(name, x, v, ind):
    """ initiaization of the second geodesic """
    t = 0
    w = (np.array(name[ind + 1]) - np.array(name[ind]) - np.inner(np.array(name[ind + 1]) - np.array(name[ind]),
                                                                  x) * x - np.inner(
        np.array(name[ind + 1]) - np.array(name[ind]), v) * v)
    w = w / mt.sqrt(np.inner(w, w))
    return t, w


def runSecond(d1, d2, name, x, v, t, w, second_stop):
    """ search of the second principal geodesics """
    while (d2 > second_stop) and (d2 > second_stop):
        t1 = t
        w1 = w
        z0 = phi(w, t, name, x, v)
        w = z0 - np.inner(z0, x) * x - np.inner(z0, v) * v
        w = w / np.inner(w, w)

        suma = 0
        sum = 0
        for i in name:
            suma = suma + epsilon(chi2(t, w, i, x, v)) * np.inner(x, i) * np.inner(v, i)
            sum = sum + epsilon(chi2(t, w, i, x, v)) * (np.power(np.inner(x, i), 2) + np.power(np.inner(v, i), 2))

        t = 0.5 * mt.atan(2 * suma / sum)
        d1 = dist1(w1 - w, t1 - t)[0]
        d2 = dist1(w1 - w, t1 - t)[1]
        y = x * mt.cos(t) + v * mt.sin(t)
    return y, w


class geopca:
    """ class containing all information about principal geodesics
    obtainedfrom a given data """

    def __init__(self, a, first_stop=0.00001, second_stop=0.0005):

        name = a

        indg = indi(name)
        indr = indu(name, indg)
        self.x, self.v = initializeFirst(name, indg, indr)
        self.x, self.v = runFirst(1., 1., name, self.x, self.v, first_stop)
        self.t, self.w = initializeSecond(name, self.x, self.v, indr)
        self.y, self.w = runSecond(1., 1., name, self.x, self.v, self.t, self.w, second_stop)

        proj_1 = projection_1(name, self.x, self.v)
        proj_2 = projection_1(name, self.y, self.w)

        self.origin = self.y
        self.angles1 = get_angles(proj_1, self.v, self.x, self.t, self.origin) * 180 / np.pi
        self.angles2 = get_angles(proj_2, self.w, self.y, 0.0, self.origin) * 180 / np.pi
        self.dataset = np.vstack((self.angles1, self.angles2)).T
        self.variance1 = variance(name, self.x, self.v)
        self.variance2 = variance(name, self.y, self.w)

    def __repr__(self):
        return str(self.variance1)+", "+str(self.variance2)

    def project(self, a):
        """
        :param a: new dataset as numpy object of angles (mapped into the sphere)
        :return: an Angles object containing projection onto first two principal component geodesics obtained by geopca with units in radians
        """
        name1 = a
        proj_11 = projection_1(name1, self.x, self.v)
        proj_21 = projection_1(name1, self.y, self.w)
        angles1 = get_angles(proj_11, self.v, self.x, self.t, self.origin) * 180 / np.pi
        angles2 = get_angles(proj_21, self.w, self.y, 0.0, self.origin) * 180 / np.pi
        c = Angles()
        c.dataset = np.vstack((angles1, angles2)).T
        c.variance1 = variance(name1, self.x, self.v)
        c.variance2 = variance(name1, self.y, self.w)
        return c

    def write(self, name):
        """ writting results into the file
        :param name: filename
        :return:  None
        """
        fil = open(name, "w")
        for i in range(self.dataset.shape[0]):
            fil.write(str(self.angles1[i]) + "," + str(self.angles2[i]) + "\n")
        fil.close()
