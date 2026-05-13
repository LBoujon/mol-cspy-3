from cspy.chem import Element
from cspy.crystal import Crystal
import logging
from os.path import join
import numpy as np
LOG = logging.getLogger(__name__)

_3ob_max_momenta_values = {
    "Br":"d",
    "C":"p",
    "Ca":"p",
    "Cl":"d",
    "F":"p",
    "H":"s",
    "I":"d",
    "K":"p",
    "Mg":"p",
    "N":"p",
    "Na":"p",
    "O":"p",
    "P":"d",
    "S":"d",
    "Zn":"d"
    }

_3ob_hubbard_derivs = {
    "Br":-0.0573,
    "C":-0.1492,
    "Ca":-0.034,
    "Cl":-0.0697,
    "F":-0.1623,
    "H":-0.1857,
    "I":-0.0433,
    "K":-0.0339,
    "Mg":-0.02,
    "N":-0.1535,
    "Na":-0.0454,
    "O":-0.1575,
    "P":-0.14,
    "S":-0.11,
    "Zn":-0.03
    }

_3ob_dispersion_parameters = {
    "3ob_standard":[0.746, 4.191, 1.0, 3.209], #parameter order: a1, a2, s6, s8
    "3ob_brandenburg":[0.841, 3.834, 1.0, 0.0] #parameter order: a1, a2, s6, s8
    }

def kpoints_per_klength(klength, kspacing=0.05):
    nk_raw = max(1, ((klength/kspacing)))
    return int(np.ceil(nk_raw))


def calc_kpoint_grid(crystal, kspacing=0.05):
    try:
        LOG.debug(f'unit cell lattice before niggli - {crystal.unit_cell.lattice}')
        crystal = crystal.standardized() 
        LOG.debug(f'unit cell lattice after niggli - {crystal.unit_cell.lattice}')
    except:
        LOG.debug('Crystal standardization failed. Keeping original crystal')
    
    k_vec_norms = [np.linalg.norm(kvec) for kvec in
                   crystal.unit_cell.reciprocal_lattice] #length of each cell vector
    nkvec = [kpoints_per_klength(knorm, kspacing) for knorm in k_vec_norms] #number of kpoints per vector length
    return nkvec


def write_gen_file(structure, output_file):
    return structure.to_gen_file(output_file)


def create_geometry_section(hsd_file, gen_filename="geometry.gen"):
    with open(hsd_file, "w") as f:
        f.writelines(
"""Geometry = GenFormat {{
    <<< {}
}}\n""".format(gen_filename))


def create_driver_section(hsd_file, max_force, max_steps,
                          out_prefix, alg="LBFGS", lattice_opt=False):
    lat = "LatticeOpt = Yes" if lattice_opt else "#"

    with open(hsd_file, "a") as f:
        f.writelines(
"""Driver = GeometryOptimization {{
    Optimizer = {}{{}}
    MovedAtoms = 1:-1
    Convergence = {{GradElem = {}}}
    MaxSteps = {}
    OutputPrefix = {}
    {}
}}\n""".format(alg, max_force, max_steps, out_prefix, lat))


def create_hamiltonian_section(hsd_file, skf_path, max_angular_momenta,
                               hubbard_derivs, disp_coeff, kpoints, scc_tol):
    
    if kpoints:
        shift_coeff = [0.5 if (float(item)%2) == 0 else 0 for item in kpoints]
        kpoints_text = """KPointsAndWeights = SupercellFolding {{                           
               {} 0 0                              
               0 {} 0
               0 0 {}
               {} {} {}                            
           }} """.format(*kpoints, *shift_coeff)
    else:
        kpoints_text = "#"
    
    with open(hsd_file, "a") as f:
        f.writelines(
"""Hamiltonian = DFTB {{
    Scc = Yes
    SlaterKosterFiles = Type2FileNames {{
        Prefix = {}/
        Separator = "-"
        Suffix = ".skf"
    }}

    ThirdOrderFull = yes
    SCCTolerance = {}
    MaxSCCIterations = 100
    HCorrection = Damping{{
        Exponent = 4.0
    }}

    Dispersion = DftD3 {{
        Damping = BeckeJohnson {{
            a1 = {}
            a2 = {}
        }}
        s6 = {}
        s8 = {}
    }}

    {}

    MaxAngularMomentum {{\n""".format(skf_path, scc_tol, 
                         *disp_coeff, kpoints_text))

        for ele, value in max_angular_momenta.items():
            f.writelines(
    """         {} = {}\n""".format(ele, value))

        f.writelines(
    """     }
    HubbardDerivs {\n""")

        for ele, value in hubbard_derivs.items():
            f.writelines(
    """        {} = {}\n""".format(ele, value))

        f.writelines("    }\n}\n")


def write_dftb_inputs(structure, working_directory, settings):
    """   
    Creates a GEN and HSD file input for a dftb+ calculation on a crystal 
    or molecule object. Currently used by DftbMinimizer to create inputs before 
    a calculation is performed.
    
    Parameters
    ----------
    structure : cspy crystal or molecule object
    settings : dictionary of keyword settings
    
    e.g.
    dftb_keywords = {
        "skf_set" = "3ob",
        "skf_path" = "User_Defined_PATH",
        "disp_coeff" = "3ob_standard",
        "groups" = 1,
        "max_force" = 0.00058, # units in Ha/Bohr - value equivalent to 0.03 eV/AA
        "alg" = "LBFGS",
        "output_prefix" = "geom.out",
        "kpoint_spacing" = 0.05,
        "scc_tolerance" = 1e-5,
        "single_point" = False,
        "fixed_lattice_opt" = False
    }
    """ 
    if isinstance(structure, Crystal):
        atom_types = set(Element[x].symbol for x in structure.unit_cell_atoms()["element"])
        kpoints=calc_kpoint_grid(structure, kspacing=settings["kpoint_spacing"])
    else:
        atom_types = set(ele.symbol for ele in structure.elements)
        kpoints = None
        settings['lattice_opt'] = False
    dict_max_momenta = {ele:_3ob_max_momenta_values[ele] for ele in atom_types}
    dict_hubbard_derivs = {ele:_3ob_hubbard_derivs[ele] for ele in atom_types}
    skf_full_path = "".join((settings["skf_path"], settings["skf_set"]))
    LOG.info("Creating dftb_in.hsd file")
    output_file = join(working_directory, "dftb_in.hsd")
    output_geo_file = join(working_directory, "geometry.gen")
    write_gen_file(structure, output_geo_file)
    create_geometry_section(output_file, gen_filename=output_geo_file)
    
    if settings["single_point"]:
       pass
    else:
        create_driver_section(output_file,
                              max_force=settings["max_force"],
                              max_steps=settings["max_steps"],
                              out_prefix=settings["output_prefix"],
                              alg=settings["alg"],
                              lattice_opt=settings["lattice_opt"])

    create_hamiltonian_section(output_file,
                               skf_path=skf_full_path,
                               scc_tol=settings["scc_tol"],
                               max_angular_momenta=dict_max_momenta,
                               disp_coeff=_3ob_dispersion_parameters[settings["disp_coeff"]],
                               hubbard_derivs=dict_hubbard_derivs,
                               kpoints=kpoints)



