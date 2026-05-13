from os.path import exists


class DmainFile:
    contents = ""

    def __init__(self, filename=""):
        self.filename = filename
        self._parse()

    def _parse(self):

        if not exists(self.filename):
            return
        with open(self.filename) as f:
            self.contents = f.read()

    def _parse_contents(self):
        pass

    @classmethod
    def from_string(cls, s):
        obj = cls()
        obj.contents = s
        obj._parse_contents()