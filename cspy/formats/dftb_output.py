import logging

LOG = logging.getLogger(__name__)


class DftbOutput():
    def __init__(self, contents, **kwargs):
        self.file_contents = contents.split("\n")
               
    def _valid(self):
        return ("Geometry converged" in self.file_contents)
    
    def _first_match(self, pattern, contents, split_index, start):
        if start == 'top':
            match = next((item for item in contents if pattern in item), None).split()[split_index]
        else:
            match = next((item for item in reversed(contents) if pattern in item), None).split()[split_index]
        return match
    
    
    def _parse_file(self):   
        expressions = {'initial_energy':['Total Energy:', 2, 'top'],
                       'final_energy':['Total Energy:', 2, 'bottom'],
                       'seed':['Chosen random seed:', -1, 'top'],
                       'final_geometry_step':['***  Geometry step:', 3, 'bottom'],
                      }
        return {key:self._first_match(value[0],
                                      self.file_contents, 
                                      value[1], 
                                      value[2]) 
                for key, value in expressions.items()}

    
#    @classmethod
#    def from_file(cls, filename):
#        return cls(contents=Path(filename).read_text().split("\n")) #test


