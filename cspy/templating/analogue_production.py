from cspy.cspympi.cspy_tasks import MinimizedStructure
from collections import deque
from cspy.db.datastore_writer import DatastoreWriter
import threading
from cspy.chem import Molecule
import numpy as np
import networkx
from scipy.sparse import csr_matrix
from networkx.algorithms import isomorphism
import numpy as np
from cspy.crystal import AsymmetricUnit
from cspy.crystal import Crystal


def get_iso_overlays(mol_1:object,mol_2:object , inds_1:list[int],inds_2:list[int],possibilities:list[tuple[list[int],list[int]]]
    ) -> list[tuple[list[int],list[int]]] :
        """Calculate all isomorphic ways of overlaying two given substructure instances and add to existing list of possible overlays

        Parameters
        ----------
        mol_1 : object
            cspy.molecule object representing one molecule

        mol_2 : object
            cspy.molecule object representing one molecule


        inds_1 : [list[int]]
            List of atom indices representing an identified instance of the key shared substructure within mol_1

        inds_2 : [list[int]]
            List of atom indices representing an identified instance of the key shared substructure within mol_2

        possibilities:  list[tuple[list[int],list[int]]] List of tuples (one per valid overlay) of lists of atom indices to overlay for that case



        Returns
        -------

           possibilities:  list[tuple[list[int],list[int]]] List of tuples (one per valid overlay) of lists of atom indices to overlay for that case
        """

        mol_1_sub = Molecule(
            elements=np.array(mol_1.elements)[inds_1],
            positions=mol_1.positions[inds_1],
            labels=mol_1.labels[inds_1],
        )
        mol_2_sub = Molecule(
            elements=np.array(mol_2.elements)[inds_2],
            positions=mol_2.positions[inds_2],
            labels=mol_2.labels[inds_2],
        )
        num_1 = mol_1_sub.atomic_numbers
        num_2 = mol_2_sub.atomic_numbers
        

        
        if mol_1_sub.bonds is None:
           mol_1_sub.guess_bonds()
        if mol_2_sub.bonds is None:
           mol_2_sub.guess_bonds()
        # create the graph matcher from the molecular connectivity
        graph_1 = networkx.from_scipy_sparse_array(csr_matrix(mol_1_sub.bonds))
        graph_2 = networkx.from_scipy_sparse_array(csr_matrix(mol_2_sub.bonds))
        gm = isomorphism.GraphMatcher(graph_1, graph_2)
        for mapping in gm.isomorphisms_iter():
            inds_2=np.array(inds_2)
            sorted_map = {i:mapping[i] for i in range(len(mol_2_sub))}
            order = list(sorted_map.values())
            reordered=inds_2[order]
            equal = np.array_equal(num_1, num_2[order])
            if equal:
               reordered=list(reordered)
               possibilities.append((inds_1,reordered))
        return possibilities


def remake_crystal(original_sg:object,new_cell:object,mols:list[object]) -> object :
        """Form crystal object from cell, spacegroup, and desired asymmetric unit molecules

        Parameters
        ----------
        original_sg : object
            spacegroup object - for templating code this is the template crystal

        mols : list[object]
            cspy.molecule object representing desired asymmetric unit molecules

        new_cell : object
            unit cell object corresponding to desired unit cell - for templating code this is a unit cell after enlarging to relieve clashes


        Returns
        -------

        new_rep: object
            cspy.crystal object formed from the cell, molecules, and spacegroup - within templating code this is an analogue after a step of clash-relief
        """
        elements = []
        labels = []
        positions=[]
        for mol in mols:
            elements += mol.elements
            labels += list(mol.labels)
            positions +=list(mol.positions)
        positions = np.asarray(positions)
        asymmetric_unit = AsymmetricUnit(
            elements, new_cell.to_fractional(positions), labels
        )
        new_rep=Crystal(new_cell,original_sg,asymmetric_unit)
        return new_rep


def reposition_centroids(new_asymm_mols: list[object],original:object) -> list[object]:
    """Shift centroid positions of one set of molecules to match the centroid positions of asymmetric unit molecules of a given crystal Note:-this is only valid for  Z'=1 currently

        Parameters
        ----------
        new_asymm mols: list[object]
            list of cspy.molecule objects

        original : object
            cspy.crystal object for which the asymmetric unit molecule (code should only be applied to Z'=1) has the desired centroid position


        Returns
        -------

        new_asymm mols: list[object]
            list of cspy.molecule objects - shifted to match the centroid position taken from the crystal structure provided
    """

    orig_asymm=original.asym_mols()
    orig_centroid=orig_asymm[0].centroid
    asym_centre=new_asymm_mols[0].centroid
    centre_shift=orig_centroid-asym_centre
    for mol in new_asymm_mols:
        mol.translate(centre_shift)
    return new_asymm_mols


def increment_uc_lengths(original_cell:object) -> tuple[float,float,float]:
        """Calculate increased unit cell lengths - based on increasng shortest unit cell length by 1 Angstrom

        Parameters
        ----------
        original_cell: unit cell object

        Returns
        -------

        tuple[float,float,float] unit cell lengths after adapatation

        """

        a=original_cell.a
        b=original_cell.b
        c=original_cell.c
        if original_cell.is_triclinic or original_cell.is_monoclinic or original_cell.is_orthorhombic:
                min_len = min([a, b, c])
                if a == min_len:
                    a += 1
                elif b == min_len:
                    b += 1
                elif c == min_len:
                    c += 1
                return a, b, c
        elif original_cell.is_tetragonal or original_cell.is_hexagonal:
                min_len = min([a, c])
                if a == min_len:
                    a += 1
                elif c == min_len:
                    c += 1
                return a, a, c
        elif original_cell.is_rhombohedral or original_cell.is_cubic:
                a += 1
                return a, a, a


def get_new_cell(original_cell:object,a:float,b:float,c:float)-> object:
    """Change cell lengths of an existing unit cell object to specified lengths

        Parameters
        ----------
        original_cell:object
                     unit cell object
        a:float
              desired cell length a
        b:float
              desired cell length b
        c:float
              desired cell length c

        Returns
        -------

        new_cell: object
            unit cell object with updated cell lengths

    """

    new_lens=np.array([a,b,c])
    old_angles=np.array([original_cell.alpha,original_cell.beta,original_cell.gamma])
    original_cell.set_lengths_and_angles(new_lens,old_angles)
    new_cell=original_cell
    return new_cell


def write_ana_to_db(analogue:object,name:str,db_name:str,targ_name:str) -> None:
    """write a given initial analogue  structure to cspy database -  supplies energies and densities of 0

        Parameters
        ----------
        analogue: object
                  cspy.crystal object to be written to database
        name: str
              name of crystal structure

        db_name: str
              name of database to write to

        targ_name: str
              name of molecule in crystal

    """

    sgnum = analogue.space_group.international_tables_number
    crys_id = name.split('.')[0]
    analogue.properties["lattice_energy"] = 0
    analogue.properties["density"] = 0
    ana_struct = MinimizedStructure(
                        molecule_id=targ_name,
                        id=crys_id,
                        spacegroup=sgnum,
                        trial_number='n/a',
                        minimization_step=0,
                        energy=analogue.properties["lattice_energy"],
                        density=analogue.properties["density"],
                        file_content=analogue.to_shelx_string(),
                        xrd=None,
                        time=0.
                     )


    structure_queue={db_name:deque()}
    structure_queue[db_name].extend([ana_struct])
    
    db_writer = DatastoreWriter(
    structure_queue,
    filename_format=f"{{db_id}}.db",
    interval=0.3,)

    db_thread = threading.Thread(
    target=db_writer.run,name=f"{db_name}-dbworker")
    db_thread.start()
    
    if db_thread is not None:
       db_writer.complete = True
       db_thread.join()


def get_shift(mol:object,original_copy:object,new_cell:object) -> object :
    """write a given initial analogue  structure to cspy database -  supplies energies and densities of 0

        Parameters
        ----------
        mol: object
             cspy.molecule object for the molecule to be moved
        
        original_copy: object
             cspy.crystal object representing the crystal before alteration to its unit cell lengths

        new_cell: object
             unit cell object with the new unit cell paramaters

        Returns
        ---------

        cart_shift: object
              np.nd.array defining the shift in cartesian co-ordinates required to move the moelcuels and maintain their fractional centroid positions within a new cell

    """

    cartesian_centre=mol.centroid
    frac=original_copy.unit_cell.to_fractional(cartesian_centre)
    new_cartesian=new_cell.to_cartesian(frac)
    cart_shift=new_cartesian-cartesian_centre
    return cart_shift

