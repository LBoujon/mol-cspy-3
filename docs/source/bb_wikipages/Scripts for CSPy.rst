.. _useful_scripts:

Useful Scripts
==============

This section will contain a set of scripts which work as standard
templates within CSPy and they are detailed where necessary. 


Niggli Cell Script
------------------

It is not unusual for DMACRYS to yield minimised unit cells that, while
physical, have very non-standard sets of lattice parameters –
particularly in cases of unconstrained angles like in
triclinic/monoclinic space groups, where cells can become very “flat”
with very large/small cell angles.

Though DMACRYS can handle these structures, a great deal of methods and
other codes (e.g. periodic DFT) may struggle, or at least behave
sub-optimally (e.g. very long, flat cell definitions will have
suboptimal k-point grids). It can therefore be helpful to obtain the
**standardised, reduced (or Niggli) unit cell**:

.. code:: python

   from cspy.crystal import Crystal
   import argparse
   from multiprocessing import Pool
   from functools import partial

   def niggli(input_file, p1):
       structure = Crystal.load(input_file)
       file_ext = input_file[-4:]
       name = input_file.split(file_ext)[0]
       niggli_structure = structure.standardized()

       if p1:
           niggli_P1 = niggli_structure.as_P1()
           niggli_P1.save('{}_p1.cif'.format(name))
       else:
           niggli_structure.save('{}.cif'.format(name))

   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument(
           "structures",
           type=str,
           nargs="+",
           help="CIF or RES files containing input structures."
                )

       parser.add_argument(
           "-j",
           "--num_threads",
           type=int,
           help="Number of parallel threads to use when doing the comparisons.",
           default=1
           )

       parser.add_argument(
           "--p1",
           action='store_true',
           help="Saves in P1 space group.",
           default=False,
           )

       args = parser.parse_args()
       structures = [item for item in args.structures]
       niggli_run = partial(niggli, p1=args.p1)

       with Pool(int(args.num_threads)) as pool:
           pool.map(niggli_run, structures)

   if __name__ == "__main__":
       main()

To run this, gather all the .res files you wish to standardize, and use the following command: 

.. code:: text

   python SCRIPT.py *.res

The script can also used with .cif files and can be multithreaded with
the ``-j NUM_OF_CORES`` flag. Additionally, it can also create P1 cells with the
``--p1`` flag.

Converting SHELX to CIF format
------------------------------

CSPy is able to convert from SHELX to CIF format and vice versa. Here is an example python script for this conversion:

.. code:: python

   from cspy.crystal import Crystal
   import argparse
   from multiprocessing import Pool

   def res_to_cif(input_file):
       structure = Crystal.load(input_file)
       name = input_file.split(".res")[0]
       structure.save('{}.cif'.format(name))

   def main():
       parser = argparse.ArgumentParser()
       parser.add_argument(
           "structures",
           type=str,
           nargs="+",
           help="RES files containing input structures to be converted."
                )

       parser.add_argument(
           "-j",
           "--num_threads",
           type=int,
           help="Number of parallel threads to use when doing the comparisons.",
           default=1
           )

       args = parser.parse_args()
       structures = [item for item in args.structures]
       with Pool(int(args.num_threads)) as pool:
           pool.map(res_to_cif, structures)

   if __name__ == "__main__":
       main()

To run this, gather all the files you wish to convert, and use the following command: 

.. code:: bash

   python SCRIPT.py *.res

The script can be multithreaded with the ``-j NUM_OF_CORES`` flag. 


Visualising the CSP landscape 
-----------------------------

Below are two python scripts that can be used to plot and visualise the CSP landscape from a resulting ``cspy-csp`` simulation.

From the SQL database
^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from cspy.db.datastore import CspDataStore
   import sys
   import matplotlib.pyplot as plt


   db = CspDataStore(sys.argv[1])
   x_and_y = [item for item in db.query("select energy, density from crystal where id like '%-3'").fetchall()] 

   plt.figure()
   x = [item[1] for item in x_and_y]
   y = [item[0] for item in x_and_y] 
   plt.scatter(x,
               [item - min(y) for item in y],
               s=10,
               edgecolor='k',
               c='b'
               )
   plt.xlabel('Density (g cm$^{-3}$)')
   plt.ylabel('Relative Energy (kJ mol$^{-1}$)')
   plt.legend()
   plt.tight_layout()
   plt.savefig('Landscape.png', dpi=600)



From an output .csv file
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   import pandas as pd
   import sys
   import matplotlib.pyplot as plt

   df = pd.read_csv(sys.argv[1])

   x = df['density']
   y = df['energy'] - min(df['energy'])
   plt.figure()
   plt.scatter(x,
               y,
               s=10,
               edgecolor='k',
               c='b'
               )
   plt.xlabel('Density (g cm$^{-3}$)')
   plt.ylabel('Relative Energy (kJ mol$^{-1}$)')
   plt.legend()
   plt.tight_layout()
   plt.savefig('Landscape.png', dpi=600)