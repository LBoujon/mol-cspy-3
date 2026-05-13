import random
from .inputs import *
from .dpca import to_cart

class Cluster:
    def __init__(self, points, topology="euclid"):
        self.points = points
        self.n = points[0].n
        self.topology = topology
        self.centroid = self.calculateCentroid()

    def __repr__(self):
        return str(self.points)

    def update(self, points):
        old_centroid = self.centroid
        self.points = points
        self.centroid = self.calculateCentroid()
        return getDistance(old_centroid, self.centroid, self.topology)

    def calculateCentroid(self):
        if self.topology == "sphere":
            centroid_coords = np.zeros(self.n)
            for p in self.points:
                centroid_coords += p.coords
            centroid_coords /= len(self.points)
            leng = np.sqrt(np.inner(centroid_coords, centroid_coords))
            centroid_coords /= leng
            return Point(centroid_coords)
        elif self.topology == "torus" or self.topology == "euclid":
            centroid_coords = np.zeros(self.n)
            for p in self.points:
              centroid_coords += p.coords
            centroid_coords /= len(self.points)
            return Point(centroid_coords)


def kmeans_solve(points, k, topology, cutoff):
    initial = random.sample(points, k)
    clusters = []
    for p in initial: clusters.append(Cluster([p], topology))
    while True:
        lists = []
        for c in clusters: lists.append([])
        for p in points:
            smallest_distance = getDistance(p, clusters[0].centroid, topology)
            index = 0
            for i in range(len(clusters[1:])):
                distance = getDistance(p, clusters[i + 1].centroid, topology)
                if distance < smallest_distance:
                    smallest_distance = distance
                    index = i + 1
            lists[index].append(p)
            p.label = index
        biggest_shift = 0.0
        for i in range(len(clusters)):
            shift = clusters[i].update(lists[i])
            biggest_shift = max(biggest_shift, shift)
        if biggest_shift < cutoff: break

    return clusters


def getDistance(a, b, topology="euclid"):
    if topology == "sphere":
        ret = np.inner(a.coords, b.coords)
        if ret > -1.0 and ret < 1.0:
            return mt.acos(ret)
        elif ret > 1.0:
            return 0.0
        else:
            return np.pi
    if topology == "torus" or topology == "euclid":
        return np.sqrt(np.sum(a.coords-b.coords)**2)

class kmeans():
    def __init__(self, b, clst, cutoff=1., topology="euclid"):
        if isinstance(b, Angles):
            self.original = b.dataset
            b.to_radians()
            self.dataset = b.dataset
        else:
            self.original = b
            self.dataset = b
        self.clst = clst
        self.cutoff = cutoff
        self.topology = topology
        if topology == "torus":
            self.points = list(map(Point, to_cart(self.dataset)))
        else:
            self.points = list(map(Point, self.dataset))
        self.clusters = kmeans_solve(self.points, self.clst, self.topology, self.cutoff)

    def write(self, file):
        fil = open(file, "w")
        for i in range(len(self.points)):
            string_to_write = str(self.points[i].label)
            for j in self.original[i]:
                string_to_write += (","+str(j))
            string_to_write += "\n"
            fil.write(string_to_write)
        fil.close()
