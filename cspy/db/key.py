class CspyKey:
    """Wrapper class around str for CSPy database
    key strings should be able to guess the format

    >>> k = CspyKey('1022200301518229')
    >>> k2 = CspyKey('1012200301518229')
    >>> k.version
    0
    >>> k['cmps']
    1
    >>> k['conf']
    22
    >>> k['sg']
    15
    >>> k['seed']
    18229
    >>> k2 == k
    False
    """

    _comp_list = {
        0: (
            ("cmps", 1, int),
            ("conf", 3, int),
            ("val", 1, int),
            ("step", 3, int),
            ("sg", 3, int),
            ("seed", None, int),
        )
    }

    def __init__(self, *args, **kwargs):
        self.s = str(*args, **kwargs)
        self._components = {}
        self._guess_version()
        self._extract_components()

    def _guess_version(self):
        self.version = 0

    def _extract_components(self):
        i = 0
        for comp, width, cast in self._comp_list[self.version]:
            if width:
                self._components[comp] = cast(self.s[i : i + width])
                i += width
            else:
                self._components[comp] = cast(self.s[i:])

    def __getitem__(self, key):
        return self._components.get(key, None)

    def __eq__(self, other):
        return self.s == other.s

    def __hash__(self):
        return hash(self.s)

    def __repr__(self):
        return self.s

    def __str__(self):
        return self.s

    def __format__(self, *args, **kwargs):
        return self.s.__format__(*args, **kwargs)

    @staticmethod
    def from_components(cmps, conf, val, step, sg, seed):

        cmps = str(cmps)
        conf = str(conf).rjust(3, "0")
        step = str(step).rjust(3, "0")
        val = str(val)
        sg = str(sg).rjust(3, "0")
        seed = str(seed)
        dbkey = cmps + conf + str(val) + step + sg + seed
        return CspyKey(dbkey)


class CspDatabaseId(str):
    _sep = "-"
    _component_order = ("name", "source", "spacegroup", "seed", "step", "other")
    _component_types = (str, str, int, int, int, str)

    def __init__(self, *args, **kwargs):
        self.s = str(*args, **kwargs)

    @property
    def tokens(self):
        return self.s.split(self._sep)

    @property
    def components(self):
        """
        >>> s = CspDatabaseId("acetic-QR-33-10011-3")
        >>> print(s.components["source"])
        QR
        """
        if not hasattr(self, "_components"):
            self._components = {
                name: t(token)
                for name, t, token in zip(
                    self._component_order,
                    self._component_types,
                    self.s.split(self._sep),
                )
            }
        return self._components

    def __getitem__(self, key):
        return self.components[key]

    @classmethod
    def from_components(cls, *components):
        return cls("-".join(str(x) for x in components))
