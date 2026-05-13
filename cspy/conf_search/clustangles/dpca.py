import numpy as np
from operator import itemgetter
import csv
import logging

LOG = logging.getLogger(__name__)

def to_cart(all_samples):
    """
    :param all_samples: dataset as numpy object with radians as units
    :return: datasets after sine and cosine transformations are applied
    """
    c = np.hstack((np.sin(all_samples), np.cos(all_samples)))
    return c


def kmo(a):
    """
    :param a: an Angle object containing dataset
    :return: KMO score for the dataset
    """
    a.to_radians()
    cart_sample = to_cart(a.dataset)
    corr = np.corrcoef(cart_sample)
    corr2 = corr ** 2
    corr_inverse = np.linalg.inv(corr)
    d = np.diag(corr_inverse)
    p2 = (-corr_inverse / np.sqrt(np.outer(d, d))) ** 2
    np.fill_diagonal(corr2, 0.0)
    np.fill_diagonal(p2, 0.0)
    kmo = np.sum(corr2) / (np.sum(corr2) + np.sum(p2))
    if kmo < 0.4999:
        print("KMO value %f is smaller than 0.5, there might be problems with sampling and result is not trustworthy" % kmo)
    else:
        print("KMO value is %f" % kmo)
    msa = np.sum(corr2, axis=0) / (np.sum(corr2, axis=0) + np.sum(p2, axis=0))
    print("msa is :", msa)


class dpca():
    """ Class describing result of dPCA application
    """
    def __init__(self, a, method="cov"):

        self.a = a
        self.a.to_radians()
        self.all_samples = to_cart(self.a.dataset)
        self.dimens = self.all_samples.shape[1]
        if method == "cov":
            self.cov_mat = np.cov(self.all_samples.T)
        else:
            self.cov_mat = np.corr(self.all_samples.T)
        self.eig_val, self.eig_vec = np.linalg.eig(self.cov_mat)
        self.eig_pairs = [[np.abs(self.eig_val[i]), (self.eig_vec[:, i])] for i in range(len(self.eig_val))]
        self.eig_pairs = sorted(self.eig_pairs, key=itemgetter(0))
        self.eig_pairs.reverse()
        eig_sort = [np.abs(self.eig_pairs[i][0]) for i in range(len(self.eig_val))]
        self.total = np.sum(np.abs(self.eig_val))
        self.percent = 100.0 * np.array(eig_sort) / self.total
        self.transformed = None

    def __repr__(self):
        return str(self.a.file)+", "+str(self.a.units)

    def project(self, number_eigen_values, b, file=None):
        """
        :param number_eigen_values: number of eigenvalues
        :param b: an Angles object containing dataset we want to project
        :param file: if output should be written to csv file, filename
        :return: numpy object containing projected dataset
        """
        assert number_eigen_values <= len(self.eig_val) and number_eigen_values > 0, "This number for eigenvalues is impossible"
        matrix_w = np.hstack([self.eig_pairs[i][1].reshape(self.dimens, 1) for i in range(number_eigen_values)])
        b.to_radians()
        sample = to_cart(b.dataset)
        self.transformed = (matrix_w.T.dot(sample.T)).T
        # print('Explained', np.sum(self.percent[:number_eigen_values]))
        if file is not None:
            with open(file, 'w') as f:
                csv.writer(f).writerows(np.real(self.transformed))
        return np.real(self.transformed)

    def scree(self):
        """
        :return: based on kink criterion a tuple (number of eigenvalues, percentage explained)
        """
        x = np.copy(np.abs(self.eig_val))
        x = (np.sort(x))[::-1]
        v = np.array([len(x) - 1, x[-1] - x[0]])
        len_v = np.sqrt((len(x) - 1) * (len(x) - 1) + (x[-1] - x[0]) * (x[-1] - x[0]))
        elbow = 0
        value = 1.0
        for i in range(1, len(x)):
            cos_angle = round(np.dot(np.array([i, x[i] - x[0]]), v) / (len_v * np.sqrt(i * i + (x[i] - x[0])*(x[i] - x[0]))), 2)
            if cos_angle <= value:
                value = cos_angle
                elbow = i
        return elbow, np.sum(self.percent[:elbow + 1])

    def advice(self, x=70):
        """
        :param x: percentage of variance at least to cover
        :return: advice about sampling quality and number of components to choose
        """
        elbow = self.scree()
        total = 0.0
        for i in range(len(self.percent)):
            total += self.percent[i]
            if total > x:
                sixty = i+1
                break

        print("""
According to kink criterion applyed to scree plot, you better choose %d component,
which explains %d percent of variability. In case you want to account at least %f percent
you need %d components""" % (elbow[0], elbow[1], float(x), sixty))


