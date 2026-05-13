def structure_generation(self, args):
    t1 = time.time()
    spacegroup, (min_seed, max_seed), name = args
    LOG.debug("min seed: %d, max seed: %d", min_seed, max_seed)
    molecules = [
        Molecule.from_xyz_string(x) for x in self.data["asymmetric_unit"]
    ]
    clg = CrystalGenerator(molecules, spacegroup)
    crystals = []
    for seed in range(min_seed, max_seed):
        generated = clg.generate(seed)
        if generated is None:
            crystals.append(None)
            continue
        res = generated.to_shelx_string(titl=str(
            CspDatabaseId.from_components(name, "QR", spacegroup, seed, 0)
        ))
        crystals.append(GeneratedStructure(
                name=name, trial_number=seed, spacegroup=spacegroup,
                file_content=res
        ))
    return CSPyTaskTag.QR, (spacegroup, crystals), time.time() - t1