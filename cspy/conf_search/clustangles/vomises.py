from .inputs import *

def besseli(r, x):
    y = (x / 2.0) * (x / 2.0)
    b0 = 1
    z1 = y / (1 + r)
    b1 = z1 * b0
    sum = 1 + b1
    for i in range(2, 100):
        z2 = y / (i * (i + r))
        b2 = z2 * b1
        sum += b2
        b1 = b2
    return ((x / 2.0) ** r) * sum / mt.gamma(1 + r)


def c(d, k):
    return (k ** ((float(d) / 2.0) - 1)) / (((np.pi) ** (float(d) / 2.0)) * besseli(((float(d) / 2.0) - 1), k))


def vonmises_dim(d):
    def vonmises(x, mu, k):
        return c(d, k) * mt.exp((np.inner(mu, x)) * k)

    return vonmises


def fit_vonmises(dataset, number):
    pi = np.array(list(range(1, number + 1)), dtype=float)
    pi /= float(np.sum(pi))
    choice = np.random.randint(dataset.shape[0], size=number)
    mu = np.copy(dataset[choice, :])
    k = np.ones(number) / float(2 * number)
    vonmises = vonmises_dim(dataset.shape[1])

    while True:

        beta = np.zeros((dataset.shape[0], pi.shape[0]))
        for i in range(dataset.shape[0]):
            l = 0.0
            for h in range(pi.shape[0]):
                l += pi[h] * vonmises(dataset[i], mu[h], k[h])
            for h in range(pi.shape[0]):
                beta[i, h] = vonmises(dataset[i], mu[h], k[h]) / l

        old_pi = np.copy(pi)
        pi = (1.0 / dataset.shape[0]) * np.sum(beta, axis=0)
        for h in range(pi.shape[0]):
            mu[h] = np.zeros(dataset.shape[1])
            for i in range(dataset.shape[0]):
                mu[h] += dataset[i] * beta[i, h]
            norm = np.linalg.norm(mu[h])
            r = norm / (number * pi[h])
            mu[h] = mu[h] / norm
            k[h] = (r * dataset.shape[1] - r * r * r) / (1 - r * r)

        if np.all(abs(pi - old_pi) / old_pi < 0.8):
            break
    return (pi, mu, k)


class vmclust():
    def __init__(self, dataset, clst):  # improve passing data
        self.clst = clst
        self.dataset = dataset
        num_points, n = dataset.shape[0], len(dataset[0])
        self.points = list(map(Point, dataset))
        self.clusters = [[] for i in range(self.clst)]
        self.parameters = fit_vonmises(dataset, self.clst)
        self.vonmises = vonmises_dim(dataset.shape[1])
        for el in self.points:
            x = []
            for k in range(self.clst):
                x.append(self.parameters[0][k] * self.vonmises(el.coords, self.parameters[1][k], self.parameters[2][k]))
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
