import logging
import numpy as np
from cspy.crystal import SpaceGroup
from cspy.chem import Element

LOG = logging.getLogger(__name__)

def parse_gen_file_content(file_contents):
    contents = file_contents.split("\n")
    num_atoms, structure_type = contents[0].split()
    symbols = [Element[x] for x in contents[1].split()]
    atoms = ["{}".format(line.split()[1])
             for line in contents[2:int(num_atoms)+2]]
    positions = ["{} {} {}".format(*pos.split()[2:])
                             for pos in contents[2:int(num_atoms)+2]]
    
    gen_dict = {"num_atoms":contents[0].split()[0]}
    gen_dict['elements'] = [symbols[int(item)-1] for item in atoms]
    gen_dict['positions'] = np.vstack([np.array([pos.split()])
                   for pos in positions]).astype(float)
    
    if structure_type == 'S':
        gen_dict['space_group'] = SpaceGroup(1)
        
        str_cell_vec = ["{} {} {}".format(*vector.split())
                       for vector in contents[-4:-1]]
        gen_dict['cell_vectors'] = np.vstack([np.array([vector.split()])
                       for vector in str_cell_vec]).astype(float)

    
    return gen_dict
