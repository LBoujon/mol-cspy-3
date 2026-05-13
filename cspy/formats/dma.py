import sys
import numpy as np
from typing import Union

def parse_dma_strings(string : Union[str,list]) -> list[dict]:
    """ Iterate over the lines from a DMA  file and return 
    its contents as a dictionary. Contents may be provided 
    as a  single string with lines seperated by '\n' or as 
    a list of strings where each element is a new line.

    Parameters
    ----------
    string : Union[str,list]
        Contents of DMA file

    Returns
    -------
    molecules_dma : list[dict]
        Each element in the list is a dictionary
         which represents a molecule from the 
         asymmetric unit.
         The dictionaries contain NEIGHCRYS labels,
         FF atom typings, coordinates, and point charges.  
    """

    if isinstance(string, list):
        contents = string
    else:
        contents = string.split('\n')

    if 'Rank 0' in contents[0]:
        rank = 0
    else:
        rank = 4
        print("Error: Only rank 0 dma files are currently supported")
        print("Exiting...")
        sys.exit()

    molecules_dma = []
    molecule = {'labels' : [],
                'atom_types' : [],
                'coords' : [],
                'charges' : []}
    for line in contents:
        line_split = line.split()
        if 'Rank' in line:
            label, x, y, z, _, _ = line_split
            label_split = label.split('_')
            if label_split[2] == '':
                atom_type = label_split[0]
            else:
                element = label_split[0]
                subtype = label_split[1]
                atom_type = element + '_' + subtype
            molecule['labels'].append(label)
            molecule['atom_types'].append(atom_type)
            molecule['coords'].append(np.asarray([float(x), float(y), float(z)]))

        elif len(line_split) == 0:
            pass

        elif '#ENDMOL' in line:
            molecules_dma.append(molecule)
            molecule = {'labels' : [],
                        'atom_types' : [],
                        'coords' : [],
                        'charges' : []}
            
        elif len(line_split) == 1:
            charges = line_split[0]
            molecule['charges'].append(float(charges))

    return molecules_dma


def parse_dma(filename : str) -> list[dict]:
    """ Read a dma file and return its contents 
    as a dictionary

    Parameters
    ----------
    filename : str
        dma filename to be read

    Returns
    -------
    molecules_dma : list[dict]
        Each element in the list is a dictionary
         which represents a molecule from the 
         asymmetric unit.
         The dictionaries contain NEIGHCRYS labels,
         FF atom typings, coordinates, and point charges.  
    """
    with open(filename, 'r') as f:
        contents = f.readlines()

    molecules_dma = parse_dma_strings(contents)

    return molecules_dma