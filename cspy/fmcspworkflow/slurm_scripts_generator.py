from cspy.templates import get_template
import subprocess

SLURM_SCRIPT_FMCSP_TEMPLATE = get_template("slurm_script_fmcsp")
VASIC_SCRIPT_TEMPLATE = get_template("vasic_script_fmcsp")
VASOPT_SCRIPT_TEMPLATE = get_template("vasopt_script_fmcsp")
VASOPT_CHECK_SCRIPT_TEMPLATE = get_template("vasopt_check_script_fmcsp")

class FMCSPScriptsClass:

    def __init__(self,slurm_data):
        self.slurm_data = slurm_data

    def slurm_script_for_fmcsp_generator(self,space_group,molecule_name):

        if space_group == "":
            crystals=""
        else:
            crystals= self.slurm_data["structures"][space_group]

        input_file = SLURM_SCRIPT_FMCSP_TEMPLATE.render(
            user=self.slurm_data["user_name"],
            name=self.slurm_data["script_name"],
            mname=self.slurm_data["molecule_name"],
            dftb_dir=self.slurm_data["dftb_dir"],
            cspy_dir=self.slurm_data["cspy_dir"],
            calc_dir=self.slurm_data["calc_dir"],
            partition=self.slurm_data["partition"],
            molecule_name=molecule_name,
            conda=self.slurm_data["conda_environment"],
            method=self.slurm_data["method"],
            basis=self.slurm_data["basis_set"],
            potential=self.slurm_data["potential"],
            sg=space_group,
            sg_list=self.slurm_data["space_group_list"],
            energy=self.slurm_data["cutoff_fmcsp_energy"],
            dftb_offset=self.slurm_data["dftb_offset_energy"],
            vasp_sp_offset=self.slurm_data["vasp_sp_offset_energy"],
            crystals=crystals,
            nodes=self.slurm_data["number_of_nodes"],
            computing=self.slurm_data["computing_time"],
            torsion_atoms=self.slurm_data["torsions_atoms_list"],
            number_of_torsions=self.slurm_data["number_of_torsions"],
            nsg=self.slurm_data["number_of_space_groups"],
            bound=self.slurm_data["bound"],
            angular_step=self.slurm_data["angular_step"],
            total_configs=self.slurm_data["total_angular_configs"],
        )

        if not space_group:
            file_name=open(self.slurm_data["script_name"]+str(molecule_name)+str(space_group)+".sh", "w+")
        else: 
            file_name=open(self.slurm_data["script_name"]+str(molecule_name)+"_"+str(space_group)+".sh", "w+")

        file_name.write(input_file)
        file_name.close()

    def VASIC_scripts_generator(self):

        input_file = VASIC_SCRIPT_TEMPLATE.render(
            user=self.slurm_data["user_name"],
            name=self.slurm_data["script_name"],
            conda=self.slurm_data["conda_environment"],
            partition=self.slurm_data["partition"],
            vasp_sp_offset=self.slurm_data["vasp_sp_offset_energy"],
            sg_list=self.slurm_data["space_group_list"],
            mname=self.slurm_data["molecule_name"],
        )

        file_name=open("VASIC"+self.slurm_data["molecule_name"]+".sh", "w+") # VAsp Sp Inputs Constructor (VASIC)
        file_name.write(input_file)
        file_name.close()

    def VASOPT_scripts_generator(self):

        input_file = VASOPT_SCRIPT_TEMPLATE.render(
            user=self.slurm_data["user_name"],
            natoms=self.slurm_data["natoms"],
            conda=self.slurm_data["conda_environment"],
            vasp_opt_offset=self.slurm_data["vasp_opt_offset_energy"],
            vasp_sp_offset=self.slurm_data["vasp_sp_offset_energy"],
            mname=self.slurm_data["molecule_name"],
        )

        file_name=open("VASOPT"+self.slurm_data["molecule_name"]+".sh", "w+") # VASp OPtimisation inputs constructor (VASOPT)
        file_name.write(input_file)
        file_name.close()

    def VASOPT_check_script_generator(self):

        input_file = VASOPT_CHECK_SCRIPT_TEMPLATE.render(
            natoms=self.slurm_data["natoms"],
            mname=self.slurm_data["molecule_name"],
        )

        file_name=open("VASOPT"+self.slurm_data["molecule_name"]+"-check.sh", "w+") # VASp OPtimisation check constructor (VASOPT-check)
        file_name.write(input_file)
        file_name.close()

    def slurm_script_submission(self,molecule_name):

        mol_submission="sbatch "+self.slurm_data["script_name"]+str(molecule_name)+".sh" 
        process_id=subprocess.run(mol_submission, capture_output=True, text=True, shell=True)
        print("main::        + ", self.slurm_data["script_name"]+str(molecule_name), "= ok ")
        return process_id.stdout.strip().split()[3]
