from ccdc import search

def make_blind_bond_mols(mol:object) -> object:
    """Take a ccdc.molecule object and make all the bonds of unknown typing

        Parameters
        ----------
        mol: object
                ccdc molecule object representing the molecule

        Returns
        -------

        mol: object
            ccdc molecule object with all bond types unknown

    """

    new_bonds=[(0,bond.atoms[0], bond.atoms[1]) for bond in mol.bonds]
    mol.remove_bonds(mol.bonds)
    mol.add_bonds(new_bonds)
    return mol


def find_substruc_cases(mol:object,substructure:object) -> list[object]:
    """Find all instances of a given substructure within a molecule

        Parameters
        ----------
        mol: object
                ccdc molecule object representing the molecule
        substructure: object
                ccdc molecule object representing the substructure

        Returns
        -------

        hits: list[object]
            list of substructure search hits

    """

    substructure_search = search.SubstructureSearch()
    max_sub=search.MoleculeSubstructure(substructure)
    sub_id = substructure_search.add_substructure(max_sub)
    hits = substructure_search.search(mol)
    return hits


def get_substructure_ids(hits: list[object]) -> list[list[int]]:
    """Find the atom indices corresponding to the found substructure for all cases in a list of substructure hits

        Parameters
        ----------
        hits: list[object]
                list of substructure search hits

        Returns
        -------

        hit_ids: list[list[int]]
            list of index lists - each list gives the ids of atoms forming the substructure

    """

    hit_ids=[]
    for hit in hits:
        matched = hit.match_atoms()
        ids = [atom.index for atom in matched]
        hit_ids.append(ids)
    return hit_ids


