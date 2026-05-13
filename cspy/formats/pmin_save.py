

class PminSave:
    def __init__(self, contents):
        self.contents = contents
        self._parse()

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            return cls(f.read())

    def _parse(self):
        all_lines = self.contents.splitlines()
        for i, line in enumerate(all_lines):
            if line.strip().startswith("The NEW"):
                lines = all_lines[i + 1 :]
                self.params = [float(x) for x in lines[0].split()]
                self.atoms = []
                for atom_line in lines[1:-1]:
                    tokens = atom_line.split()
                    if len(tokens) < 5:
                        tmp_1 = tokens[-4][:3]
                        tmp_2 = tokens[-4][3:]
                        tokens = [tmp_1, tmp_2, *tokens[1:]]
                    self.atoms.append(
                        (
                            " ".join(tokens[:-4]),
                            int(tokens[-4]),
                            [float(x) for x in tokens[-3:]],
                        )
                    )
                self.xtra = [float(x) for x in lines[-1].split()[-3:]]
