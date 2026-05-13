from .inputs import *

def kummer(a, b, z, tol):
    """ obtaining values of confluent hypergeometric function by Tailor series"""
    a1 = 1.0
    b1 = 1.0
    for j in range(500):
        a1 = ((float(a) + j) / (float(b) + j)) * (float(z) / (j + 1)) * a1
        b1 = b1 + a1
        if abs(a1) / abs(b1) < tol:
            break
    return b1


def kummer_calc(a, b, z, tol):
    a1 = 1.0
    b1 = 0.0
    for j in range(500):
        a1 = ((float(a) + j) / (float(b) + j)) * (float(z) / (j + 1)) * a1
        b1 = b1 + a1 * (j + 1) / z
        if abs(a1) / abs(b1) < tol:
            break
    return b1


def g(a, b, z, tol):
    return kummer_calc(a, b, z, tol) / kummer(a, b, z, tol)


def watson_dim(d):
    "watson function for given dimension"

    def watson(x, mu, koef):
        return mt.exp(float(koef) * (np.inner(mu, x)) * (np.inner(mu, x))) * mt.gamma(float(d) / 2) / (
            kummer(0.5, d / 2, koef, 0.0000001) * np.power(2.0 * np.pi, d / 2))

    return watson


class Point:
    def __init__(self, coords, label=0, reference=None):
        self.coords = coords
        self.label = label  # to which cluster it belongs
        self.n = len(coords)
        self.reference = reference

    def __repr__(self):
        return str(self.coords) + ", " + str(self.label)  # add to other classes in the code


def fit_watson(dataset, number, p):  # do I really need p?
    pi = np.random.uniiform(0, 1.0, number)
    pi /= float(np.sum(pi))
    choice = np.random.randint(len(dataset), size=number)
    mu = np.copy(dataset[choice, :])
    k = np.ones(number) / float(2 * number)
    watson = watson_dim(p)

    while True:
        beta = np.zeros((dataset.shape[0], pi.shape[0]))
        for i in range(dataset.shape[0]):
            l = 0.0
            for j in range(pi.shape[0]):
                l += pi[j] * watson(dataset[i], mu[j], k[j])
            for j in range(pi.shape[0]):
                beta[i, j] = pi[j] * watson(dataset[i], mu[j], k[j]) / l

        S = []
        l = np.sum(beta, axis=0)
        for j in range(pi.shape[0]):
            S1 = np.zeros((pi.shape[0], pi.shape[0]))
            for i in range(dataset.shape[0]):
                S1 += beta[i, j] * (np.outer(dataset[i], dataset[i])) / l[j]
            S.append(S1)

        eigen_pairs = []
        for j in range(pi.shape[0]):
            eig_val, eig_vec = np.linalg.eig(S[j])
            eig_pairs = [[eig_val[i], eig_vec[:, i]] for i in range(len(eig_val))]
            eig_pairs.sort()
            eigen_pairs.append(eig_pairs)

        old_pi = np.copy(pi)

        for j in range(pi.shape[0]):
            if k[j] > 0:
                mu[j] = eigen_pairs[j][0][1]
            else:
                mu[j] = eigen_pairs[j][-1][1]
            r = np.dot(mu[j].T, np.dot(S[j], mu[j]))
            k[j] = 1.0 / g(0.5, p / 2.0, r, 0.0000000005)

        pi = np.sum(beta, axis=0) / dataset.shape[0]
        if np.all(abs(pi - old_pi) / old_pi < 0.0000001):
            break

    return (pi, mu, k)


def fit_watson_hard(dataset, number, p):
    pi = np.array(list(range(1, number + 1)), dtype=float)
    pi /= float(np.sum(pi))
    choice = np.random.randint(len(dataset), size=number)
    mu = np.copy(dataset[choice, :])
    k = np.ones(number) / float(2 * number)
    watson = watson_dim(p)

    while True:
        beta = np.zeros((len(dataset), pi.shape[0]))
        for i in range(len(dataset)):
            wp = np.array([np.log(watson(dataset[i], mu[j], k[j])) for j in range(pi.shape[0])])
            for j in range(pi.shape[0]):
                beta[i, (np.log(pi) + wp).argmax(axis=0)] = 1.0

        S = []
        l = np.sum(beta, axis=0)
        print(l)
        for j in range(pi.shape[0]):
            S1 = np.zeros((pi.shape[0], pi.shape[0]))
            for i in range(dataset.shape[0]):
                S1 += beta[i, j] * (np.outer(dataset[i], dataset[i])) / l[j]
            S.append(S1)

        eigen_pairs = []
        for j in range(pi.shape[0]):
            eig_val, eig_vec = np.linalg.eig(S[j])
            eig_pairs = [[eig_val[i], eig_vec[:, i]] for i in range(len(eig_val))]
            eig_pairs.sort()
            eigen_pairs.append(eig_pairs)

        old_pi = np.copy(pi)

        for j in range(pi.shape[0]):
            if k[j] > 0:
                mu[j] = eigen_pairs[j][0][1]
            else:
                mu[j] = eigen_pairs[j][-1][1]
            r = np.dot(mu[j].T, np.dot(S[j], mu[j]))
            k[j] = 1.0 / g(0.5, p / 2.0, r, 0.0000000005)

        pi = np.sum(beta, axis=0) / dataset.shape[0]
        if np.all(abs(pi - old_pi) / old_pi < 0.001):
            break

    return pi, mu, k


class wclust():
    def __init__(self, dataset, clst):
        self.clst = clst
        self.dataset = dataset
        num_points, n = np.array(dataset).shape[0], len(dataset[0])
        self.points = list(map(Point, dataset))
        self.clusters = [[] for i in range(self.clst)]
        self.parameters = fit_watson_hard(np.array(dataset), self.clst, n)
        self.watson = watson_dim(n)
        for el in self.points:
            x = []
            for k in range(self.clst):
                x.append(self.parameters[0][k] * self.watson(el.coords, self.parameters[1][k], self.parameters[2][k]))
            x = np.array(x)
            index = x.argmax(axis=0)
            el.label = index
            self.clusters[index].append(el)

    def write(self, file):
        fil = open(file, "w")
        for i in range(len(self.points)):
            string_to_write = str(self.points[i].label)
            for j in self.dataset[i]:
                string_to_write += (","+str(j))
            string_to_write += "\n"
            fil.write(string_to_write)
        fil.close()
