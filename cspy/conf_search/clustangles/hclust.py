from .inputs import *


class node:
    def __init__(self, vec, left=None, right=None, distance=0.0, id=None, count=1):
        self.left = left
        self.right = right
        self.vec = vec
        self.id = id
        self.distance = distance
        self.count = count  # weighted average in future


def L2dist(v1, v3):
    return np.linalg.norm(v1 - v3)


def circdist(v1, v2):
    ret = np.inner(v1, v2)
    if ret > -1.0 and ret < 1.0:
        return mt.acos(ret)
    elif ret > 1.0:
        return 0.0
    else:
        return np.pi


class hclust():
    def __init__(self, b, topology="euclid"):
        if isinstance(b, Angles):
            self.original = b.dataset
            b.to_radians()
            self.dataset = b.dataset
        else:
            self.original = b
            self.dataset = b
        distances = {}
        currentclustid = 1
        self.topology = topology
        if self.topology == "sphere":
            def distance(v1, v2):
                return circdist(v1, v2)
        else:
            def distance(v1, v2):
                return L2dist(v1, v2)
        self.clust = [node(np.array(self.dataset[i]), id=i) for i in range(len(self.dataset))]
        n = len(self.clust[0].vec)
        while len(self.clust) > 1:
            lowestpair = (0, 1)
            closest = distance(self.clust[0].vec, self.clust[1].vec)

            for i in range(len(self.clust) - 1):
                for j in range(i + 1, len(self.clust)):
                    if (self.clust[i].id, self.clust[j].id) not in distances:
                        distances[(self.clust[i].id, self.clust[j].id)] = distance(self.clust[i].vec, self.clust[j].vec)
                    d = distances[(self.clust[i].id, self.clust[j].id)]
                    if d < closest:
                        closest = d
                        lowestpair = (i, j)

            if self.topology == "sphere":
                mergevec = [(self.clust[lowestpair[0]].vec[i] + self.clust[lowestpair[1]].vec[i]) / 2.0 for i in
                            range(n)]
                leng = np.sqrt(np.inner(mergevec, mergevec))
                mergevec = np.array(mergevec) / leng
            else:
                mergevec = [(self.clust[lowestpair[0]].vec[i] + self.clust[lowestpair[1]].vec[i]) / 2.0 for i in
                            range(n)]

            newcluster = node(np.array(mergevec), left=self.clust[lowestpair[0]], right=self.clust[lowestpair[1]],
                              distance=closest, id=currentclustid)
            currentclustid -= 1
            del self.clust[lowestpair[1]]
            del self.clust[lowestpair[0]]
            self.clust.append(newcluster)
            self.result = self.clust[0]

    def write(self, dist, file):
        clust = clusters(self.result, dist)
        fil = open(file, "w")
        for i in range(len(clust)):
            x = get_cluster_elements(clust[i])
            for j in x:
                string_to_write = str(i)
                for el in self.dataset[j]:
                    string_to_write += ("," + str(el))
                string_to_write += "\n"
                fil.write(string_to_write)


def clusters(clust, dist):
    if clust.distance < dist:
        return [clust]
    else:
        cl = []
        cr = []
    if clust.left is not None:
        cl = clusters(clust.left, dist=dist)
    if clust.right is not None:
        cr = clusters(clust.right, dist=dist)
    return cl + cr


def get_cluster_elements(clust):
    if clust.id > 0:
        return [clust.id]
    else:
        cl = []
        cr = []
        if clust.left is not None:
            cl = get_cluster_elements(clust.left)
        if clust.right is not None:
            cr = get_cluster_elements(clust.right)
        return cl + cr

