import warnings
import logging
import os
import sys
import time
import math
import statistics
from cspy.crystal import AsymmetricUnit
from collections import deque, namedtuple
import pandas as pd
import numpy as np
import pandas as pd
from copy import deepcopy
from cspy.db import CspDataStore
from cspy.apps.setup_app import CspyApp, add_DMACRYS_arguments, add_common_arguments, add_minimisation_arguments
from cspy.chem import Molecule
from cspy.crystal import Crystal
import ccdc
from cspy.templating.analogue_production_ccdc import make_blind_bond_mols,find_substruc_cases, get_substructure_ids
from cspy.templating.analogue_production import get_iso_overlays,remake_crystal,reposition_centroids,increment_uc_lengths,get_new_cell,get_shift,write_ana_to_db
from zipfile import ZipFile
from cspy.chem import Molecule
import networkx
from scipy.sparse import csr_matrix
from networkx.algorithms import isomorphism



LOG = logging.getLogger(__name__)


class TemplatingGenerator:
     """Class for generating intial analogues from a template set.

     Parameters
     ----------
     targ_mol : str
        molecule file name for the target molecule xyz or mol2 format

     temp_mol : str
        molecule file name for the template molecule xyz or mol2 format

     template_structures : str
        file name for the set of template crystal structures - cspy database or zip file of res files

     out_db_name : str
        desired name (without the .db suffix) for the database of analogues

     conn: bool
        whether the maximum substructure to be identified must be able to be represented by a single connected graph

     bond_check: bool
        whether the substructure detection should be concious of bond-type
        
     """

     def __init__(
        self,
        targ_mol: str,
        temp_mol: str,
        template_structures: str,
        out_db_name: str,
        conn: bool,
        bond_check: bool) -> None:
        

        self.template_structures = template_structures
        self.out_db_name = out_db_name 
        self.targ_mol = Molecule.load(targ_mol)
        self.temp_mol = Molecule.load(temp_mol)
        self.conn = conn
        self.bond_check = bond_check
        self.targ_name = targ_mol.split('.')[0]


        if targ_mol.split('.')[-1] == 'mol2':
           mol_reader=ccdc.io.MoleculeReader(targ_mol)
           self.targ_mol2 = mol_reader[0]
        else:
           #convert to mol2
           LOG.warning('Conversion to mol2 will be performed. Conversion can cause issues on occassion, if results are unexpected, check this conversion')
           self.targ_mol.to_mol2_file(targ_mol.split('.')[0] + '.mol2')
           mol_reader=ccdc.io.MoleculeReader(targ_mol.split('.') + '.mol2')
           self.targ_mol2 = mol_reader[0]
        if temp_mol.split('.')[-1] == 'mol2':
           mol_reader=ccdc.io.MoleculeReader(temp_mol)
           self.temp_mol2 = mol_reader[0]

        else:
           #convert to mol2
           LOG.warning('Conversion to mol2 will be performed. Conversion can cause issues on occassion, if results are unexpected, check this conversion')
           self.temp_mol.to_mol2_file(temp_mol.split('.')[0] + '.mol2')
           mol_reader=ccdc.io.MoleculeReader(temp_mol.split('.' + '.mol2'))
           self.temp_mol2 = mol_reader[0]

     def get_shared_substructure_instances(self) -> tuple[list[list[int]],list[list[int]]]:
          """Find the maximum shared substructure between two molecules and return instances of that substructure within each molecule.

          Returns
          ----------
          temp_mol_hits : list[list[int]]
            List of lists, each holding the atom indices rperesenting an identified instance of the key shared substructure within the template molecule

          targ_mol_hits : list[list[int]]
            List of lists, each holding the atom indices rperesenting an identified instance of the key shared substructure within the target molecule
          """

          
          #find maximum shared substructure
          sub = ccdc.descriptors.MolecularDescriptors.MaximumCommonSubstructure()
          sub.settings.check_bond_type=self.bond_check
          sub.settings.connected=self.conn
          max_substruc = sub.search(self.temp_mol2,self.targ_mol2)
          
          #make molecule object corresponding to substructure
          substructure = self.temp_mol2

          temp_mol2_bkp = self.temp_mol2.copy()
          
          sub_atoms=[atom[0] for atom in max_substruc[0]]
          sub_bonds=[bond[0] for bond in max_substruc[1]]

          ex_bonds=[bond for bond in substructure.bonds if bond not in sub_bonds]
          ex_atoms=[ato for ato in substructure.atoms if ato not in sub_atoms]

          substructure.remove_bonds(ex_bonds)
          substructure.remove_atoms(ex_atoms)

          self.temp_mol2=temp_mol2_bkp
          if not self.bond_check:
             #change all bond types to unknown to make blind to bond type     
             temp_mol=make_blind_bond_mols(self.temp_mol2)
             targ_mol=make_blind_bond_mols(self.targ_mol2)
             substructure=make_blind_bond_mols(substructure)
          
          #search for substructure in both template and target molecules
          temp_hits=find_substruc_cases(temp_mol,substructure)
          targ_hits=find_substruc_cases(targ_mol,substructure)

          #get atom indices from susbtructure hits
          temp_mol_hits=get_substructure_ids(temp_hits)
          targ_mol_hits=get_substructure_ids(targ_hits)

          return temp_mol_hits,targ_mol_hits
     
     def get_substructure_mappings(self,temp_mol_hits:list[list[int]],targ_mol_hits:list[list[int]]) -> list[tuple[list[int],list[int]]]:
         """Get sets of atom indices to overlay to cover all valid substructure overlays.

        Parameters
        ----------
        temp_mol_hits : list[list[int]]
            List of lists, each holding the atom indices rperesenting an identified instance of the key shared substructure within the template molecule

        targ_mol_hits : list[list[int]]
            List of lists, each holding the atom indices rperesenting an identified instance of the key shared substructure within the target molecule

        Returns
        -------
        
           possibilities:  list[tuple[list[int],list[int]]] List of tuples (one per valid overlay) of lists of atom indices to overlay for that case
        """
         possibilities=[]
         combos=[]
         for i in temp_mol_hits:
             for j in targ_mol_hits:
                 combos.append((i,j))
         for combo in combos:
             possibilities=get_iso_overlays(self.temp_mol,self.targ_mol,combo[0],combo[1],possibilities)
         return possibilities

     def detect_clash(self,analogue:object) -> bool:
        """Check for molecular clashes in initial analogue.

        Parameters
        ----------
        analogue : object cspy.crystal object representing the analogue formed

        Returns
        -------

        clash: bool whether or not molecular clashes have been detected
        """
        if hasattr(analogue, "_unit_cell_molecules"):
            delattr(analogue, "_unit_cell_molecules")
        exp_len=len(self.targ_mol)
        clash=False
        mols=analogue.unit_cell_molecules(conn_tolerance=0.5)
        for mol in mols:
             if len(mol)!= exp_len:
                clash=True
                break
             iso=mol.has_similar_molecular_graph_to(self.targ_mol)
             if not iso:
                clash=True
                break
        return clash

     def relieve_clash(self,template_crystal:object,analogue:object,analogue_mols:list[object]) -> tuple[object,bool]:
         """Adjust initial analogue to remove any molecular clashes

         Parameters
         ----------
         template_crystal : object cspy.crystal object representing the template crystal used

         analogue : object cspy.crystal object representing the analogue formed

         analogue_mols: list[object] list of cspy.molecule objects corresponding to the analogue molecules as overlaid with the template crystal asymmetric unit


         Returns
         -------
         replaced : object cspy.crystal object representing the analogue formed after alteration to remove classes

         clash: bool whether or not molecular clashes have been detected
         """

         template_sg=template_crystal.space_group
         clash=self.detect_clash(analogue)
         count=0
         while clash and count < 60:
               ana_copy=deepcopy(analogue) #It may be possible to remove this copying in further development, but it is not unduly costly
               ana_cell=analogue.unit_cell
               new_lengths=increment_uc_lengths(ana_cell)
               new_cell=get_new_cell(ana_cell,new_lengths[0],new_lengths[1],new_lengths[2])
               for mol in analogue_mols:
                  shift=get_shift(mol,ana_copy,new_cell)
                  mol.translate(shift)
               new_rep=remake_crystal(template_sg,new_cell,analogue_mols)
               analogue=deepcopy(new_rep)
               clash=self.detect_clash(analogue)
               count +=1
         return analogue,clash

     def make_one_analogue(self,overlay_case:tuple[list[int],list[int]],switch_name:str,case_count:int,template_crys:object) -> None:
         """Form an initial analogue of a single template crystal and single valid substructure overlay

         Parameters
         ----------
         overlay case: tuple[list[int],list[int]] tuple of lists of atom indices to overlay to achieve the relevant substructure overlay

         switch_name : str string, used for naming analogues, describes the templating case i.e <new_mol>_in_<template_crystal>

         case_count: int numeric identifier for the valid overlay instance being applied

         template_crys : object cspy.crystal object representing the template crystal used
        
    
         """


         temp_substruc=overlay_case[0]
         targ_substruc=overlay_case[1]
         try:
            #make copy of template crystal as don't want to alter original structure itself
            fresh_template=deepcopy(template_crys)
            #attempt replace by overlay to orient (and roughly position) centroids
            replacement = fresh_template.replace_molecules_by_substructure_overlay(self.targ_mol,temp_substruc,targ_substruc)
            mols=replacement[0]
            rmsd=replacement[1][0]#assumes prime 1
            if rmsd > 2.0:
                LOG.warning('RMSD failure for %s version %d skipping and continuing',switch_name,case_count)
         
            #shift centroids and reconstruct resulting crystal
            reposition_centroids(mols,fresh_template)
            template_sg=fresh_template.space_group
            template_cell=fresh_template.unit_cell
            replaced=remake_crystal(template_sg,template_cell,mols)
            #detect and relive any clashes - must go through 'reconstruction process' rather than simply shifting unit cell mol positions as they will be incorrectly detected in the event of a clash
            clash=self.detect_clash(replaced)
            if clash:
               LOG.info('Intermolecular clash detected for %s version %d. Attempting to relieve this.',switch_name,case_count)
               relief=self.relieve_clash(fresh_template,replaced,mols)
               if not relief[1]:#checking clash status after relief attempt
                  replaced=relief[0]
                  name=switch_name + '_v' + str(case_count) + '.res'
                  write_ana_to_db(replaced,name,self.out_db_name,self.targ_name)
               else:
                  LOG.warning('Failure during construction and clash relief for %s version %d It is likely clashes could not be relieved. Skipping and continuing',switch_name,case_count)

            

            else:
               replaced=deepcopy(replaced)
               name=switch_name + '_' +str(case_count) + '.res'
               write_ana_to_db(replaced,name,self.out_db_name,self.targ_name)
         except:
            LOG.warning('Unknown failure for %s version %d  skipping and continuing',switch_name,case_count)

     def make_all_analogues(self,possibilities:list[tuple[list[int],list[int]]]) -> None:
         """Form all valid initial analouges for a given template set and set of possible overlays

         Parameters
         ----------
         possibilties: list[tuple[list[int],list[int]]] list of tuple of lists of atom indices to overlay - each tuple corresponds to one valid substructure overlay

         """

         #loop over all templates, extract/load in if necessary then create all analogues of that template
         if self.template_structures.split('.')[1] == 'zip':
            with ZipFile(self.template_structures) as f:
                  for temp_struc in f.namelist():
                      f.extract(temp_struc)
                      temp_crystal=Crystal.load(temp_struc)
                      zprime = len(temp_crystal.asym_mols())
                      if zprime != 1:
                          LOG.warning('%s is not a Z prime 1 crystal. Cannot proceed in this version of the code. Skipping and continuing',temp_struc.split('.')[0])
                          continue
                      case_count=1
                      for overlay_case in possibilities:
                          switch_name = self.targ_name + '_in_' + temp_struc.split('.')[0] 
                          self.make_one_analogue(overlay_case,switch_name,case_count,temp_crystal)
                          case_count+=1
                      os.remove(temp_struc)
         elif self.template_structures.split('.')[1] == 'db':
           unique_query_text= (
           "select crystal.*, minimization_step, "
           "trial_number, minimization_time, metadata "
           "from crystal join "
           "(select distinct unique_id from equivalent_to) "
           "on crystal.id = unique_id join "
           "(select distinct(id), minimization_step, "
           "trial_number, minimization_time, metadata from trial_structure) T "
           " on T.id = crystal.id "
           )

           all_query_text  = (
           "select crystal.*, minimization_step, "
           "trial_number, minimization_time, metadata "
           "from crystal join "
           "(select distinct(id), minimization_step, "
           "trial_number, minimization_time, metadata from trial_structure) T "
           "on crystal.id = T.id "
           )

                           
           ds = CspDataStore(self.template_structures)
           unique_data = pd.read_sql(unique_query_text, ds.connection)
           structure_info = unique_data.pop("file_content")
           if len(structure_info) ==0:
                LOG.warning("Template Database %s contains no unique structures. Forming analogues of all structures instead.", self.template_structures)
                all_data = pd.read_sql(all_query_text, ds.connection)
                structure_info = all_data.pop("file_content")
           for crys in structure_info:
               temp_crystal = Crystal.from_shelx_string(crys)
               case_count=1
               zprime = len(temp_crystal.asym_mols())
               if zprime != 1:
                  LOG.warning('%s is not a Z prime 1 crystal. Cannot proceed in this version of the code. Skipping and continuing',temp_crystal.titl.split('_')[0])
                  continue

               for overlay_case in possibilities:
                          switch_name = self.targ_name + '_in_' + temp_crystal.titl.split('_')[0]
                          self.make_one_analogue(overlay_case,switch_name,case_count,temp_crystal)
                          case_count+=1

     def run(self) -> None:
        """ Form all valid analogues of a template set for a given target molecule
        """
        temp_mol_hits,targ_mol_hits = self.get_shared_substructure_instances()
        possibilities = self.get_substructure_mappings(temp_mol_hits,targ_mol_hits)
        self.make_all_analogues(possibilities)


def main():
    import argparse
    from cspy.configuration import CONFIG
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
    "--out_db_name",
    type=str,
    help="The name of the output SQLite database of unminimised analogues",
    default="ana_database"
    )
   
    parser.add_argument(
    "--template_structures",
    type=str,
    help="The name of the zip file or cspy database containing template structures"
    )

    parser.add_argument(
    "--targ_mol",
    type=str,
    help="The name of the file for the target molecule"
    )
    
    parser.add_argument(
    "--temp_mol",
    type=str,
    help="The name of the file for the template molecule."
    )

    parser.add_argument(
    "--conn",
    type=bool,
    default=True,
    help="Whether the substructure has to be connected"
    )
    
    parser.add_argument(
    "--bond_check",
    type=bool,
    default=False,
    help="Whether the substructure check should pay attention to bond type"
    )

    args = parser.parse_args()

    #Make analogues
    
    Templater=TemplatingGenerator(args.targ_mol,args.temp_mol,args.template_structures,args.out_db_name,args.conn,args.bond_check)
    Templater.run()


if __name__ == "__main__":
    main()
