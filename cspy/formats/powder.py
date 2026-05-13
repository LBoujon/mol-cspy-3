import numpy as np
try:
    from scipy.integrate import simps
except ImportError:
    from scipy.integrate import simpson as simps


class PowderAscii:
    def __init__(self, contents, bins=1000, **kwargs):
        arr = np.array([[float(x) for x in line.strip().split()] for line in contents])
        xrd = np.histogram(
            arr[:, 0], weights=arr[:, 1], bins=bins, density=False, **kwargs
        )
        self.intensities, self.two_theta = xrd
        self.two_theta = 0.5 * (self.two_theta[1:] + self.two_theta[:-1])

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            contents = f.read()
        return PowderAscii(contents)


class PowderSietronics:
    def __init__(self, contents, **kwargs):
        self.lines = contents.split("\n")
        self._parse()

    def _parse(self):
        self.header = []

        xrd = []
        line_iter = iter(self.lines)
        self.header = [line for i, line in zip(range(10), line_iter)]
        self.intensities = np.array(
            [[float(x) for x in line.strip().split()] for line in line_iter if line]
        ).reshape(-1)

        start = float(self.header[1])
        end = float(self.header[2])
        step = float(self.header[3])
        self.two_theta = np.arange(start, end, step)

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            contents = f.read()
        return PowderSietronics(contents)


def compare_row(row, powder, method="cdtw"):
    import cspy.ml.distance.cdtw as cdtw

    key, density, lattice_energy, xrd = row
    if xrd:
        xrd = np.frombuffer(xrd, dtype=np.float64)
        xrd = np.histogram(
            np.arange(0, 50, 0.02), weights=xrd, bins=1000, density=False, range=(5, 40)
        )[0]
        xrd = xrd / np.max(xrd) * np.max(powder.intensities)
        res = cdtw.cdtw_distance(powder.intensities, xrd, 10)
    else:
        res = 1000.0
    return (res, density, lattice_energy)


def search_db(filename, powder, pos):
    from tqdm import tqdm
    from cspy.db import DataStore, JobSerializer
    import os

    results = {}
    dstore = DataStore(filename)
    rows = dstore.query("select old_key, density, lattice_energy, xrd from structures")
    for row in tqdm(
        rows,
        desc="{:>30s}".format(os.path.basename(filename).split(".")[0]),
        unit=" structures",
        unit_scale=True,
        position=pos,
    ):
        r, d, e = compare_row(row, powder)
        results[row[0]] = (r, d, e, filename)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--xrd", type=str, required=True, help="xrd pattern files")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="number of threads")

    parser.add_argument("databases", nargs="+", type=str, help="databases")

    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    powder = PowderAscii(args.xrd, bins=1000, range=(5, 40))
    results = {}
    powder.intensities = powder.intensities / powder.intensities.max()
    with ThreadPoolExecutor(args.jobs) as e:
        futures = {
            e.submit(search_db, filename, powder, i % args.jobs + 1)
            for i, filename in enumerate(args.databases, start=0)
        }
        for future in tqdm(
            as_completed(futures),
            position=0,
            unit="files",
            total=len(args.databases),
            desc="Searching",
        ):
            results.update(future.result())

    results = sorted(results.items(), key=lambda x: x[1][0])
    top = results[:25]
    for x in range(args.jobs):
        print()
    print()
    print("Top 25 Results: ({} total)".format(tqdm.format_sizeof(len(results))))
    print("\n".join("{}: {}".format(x, y[:3]) for x, y in top))
    # for x in top:
    #    fig, ax = plt.subplots()
    #    fig.set_size_inches(10, 6)
    #    ax.plot(powder.two_theta, powder.intensities, label='Query')
    #    ax.plot(np.linspace(0, 50, x[1][3].shape[0]), x[1][3], label=x[0])
    #    plt.legend()
    #    plt.savefig('{}.png'.format(x[0]), dpi=300, bbox_inches='tight')
    with open("matches.csv", "w") as f:
        f.write("key,dist,density,lattice energy,filename\n")
        for k, v in results:
            f.write("{},{},{},{},{}\n".format(k, *v))


if __name__ == "__main__":
    main()
