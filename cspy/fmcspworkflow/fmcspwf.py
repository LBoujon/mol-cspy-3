
__author__  = "Dr Ramon Cuadrado"
__email__   = "ramon.cuadrado@gmail.com"

from cspy.crystal import Crystal
import subprocess
import pandas as pd
import toml
import os
from cspy.db.datastore import CspDataStore
from collections import Counter
from cspy.configuration import COMMON_SAMPLING_SETTINGS
from cspy.fmcspworkflow import FMCSPScriptsClass

def dftb_toml_file_generator():

#-- Opening file 
    file_name=open("cspy.toml", "w+")

#-- Main cspy.toml dftb options
    file_name.write("[[csp_minimization_step]]\n")
    file_name.write("kind = \"dftb\"\n")
    file_name.write("\n")
    file_name.write("[dftb]\n")
    file_name.write("skf_set = \"3ob-3-1\"\n")
    file_name.write("skf_path = \"/scratch/rcdb1c20/home2-dir/Liverpool/DFTB-optimisations/\"\n")
    file_name.write("disp_coeff = \"3ob_brandenburg\"\n")
    file_name.write("groups = 1\n")
    file_name.write("max_steps = 2500\n")
    file_name.write("max_force = 0.01\n")
    file_name.write("alg = \"LBFGS\"\n")
    file_name.write("output_prefix = \"geom.out\"\n")
    file_name.write("kpoint_spacing = 0.05\n")
    file_name.write("scc_tol = 1e-4\n")
    file_name.write("timeout = 160800\n")
    file_name.write("single_point =  false\n")
    file_name.write("fixed_lattice_opt = false\n")
    file_name.write("lattice_opt = true\n")

#-- Closing file 
    file_name.close()

def dftb_input_database_generator(offset, main_name, sg_list):

    if not isinstance(sg_list, list):
        sg_list = [sg_list]

    dftb_toml_file_generator()
    if len(sg_list) > 1:
        input_files=['structures.csv']
    else:
        input_files=['structures-'+str(sg_list[0])+'.csv']

#-- Getting the overall GM 
    all_sgs = pd.concat((pd.read_csv(i) for i in input_files)).reset_index(drop=True)
    all_sgs = all_sgs.drop_duplicates()
    gm_value = min(all_sgs['energy'])
    global_e_limit = gm_value + offset

    for sg in sg_list:
        sg_i = pd.concat((pd.read_csv(i) for i in ['structures-'+str(sg)+'.csv'])).reset_index(drop=True)
        sg_i = sg_i.drop_duplicates()
        sg_i_offset = sg_i.loc[(sg_i['energy'] <= global_e_limit)]
        if len(sg_i_offset) == 1:
            unique_id_list = tuple([item for item in sg_i_offset['id']])
            unique_id_list = 2*unique_id_list
        else:
            unique_id_list = tuple([item for item in sg_i_offset['id']])
        db = CspDataStore(str(main_name)+"-"+str(sg)+".db")
        other = CspDataStore(f'dftb_input_{sg}_{offset}.db')
        db.query(f"attach 'dftb_input_{sg}_{offset}.db' as tmp")
        db.query(
            f'insert or replace into tmp.crystal select crystal.* \
              from crystal where crystal.id in {unique_id_list}')
        db.query('insert or replace into tmp.descriptor select descriptor.* \
                  from descriptor where descriptor.id in (select id from tmp.crystal)')
        db.query('insert or replace into tmp.trial_structure select trial_structure.* \
                  from trial_structure where trial_structure.id in (select id from tmp.crystal)')
        db.commit()
        db.query("detach tmp")

        a = other.query("select trial_number from trial_structure")
        seeds = sorted(set([item[0] for item in a.fetchall()]))
        other.close()

def niggli(input_file):
    try:
        structure = Crystal.load(input_file)
        file_ext = input_file[-4:]
        name = input_file.split(file_ext)[0]
        niggli_structure = structure.standardized()
        niggli_structure.save('{}.cif'.format(name))
    except Exception as e:
        print(f'{input_file} has failed - {e}')

def VASP_sp_script_generator(script_name,partition,vasp_sp_offset,molecule_name):

#-- Opening file 
    file_name=open("SP"+script_name+".sh", "w+")

    file_name.write("#!/bin/bash\n")
    file_name.write("#SBATCH --job-name  SP"+script_name+"\n")
    file_name.write("#SBATCH --time=12:00:00\n")
    file_name.write("#SBATCH --nodes=2\n")
    file_name.write("#SBATCH --partition="+partition+"\n")
    file_name.write("#SBATCH --ntasks-per-node=40\n")
    file_name.write("#SBATCH --ntasks=80\n")
    file_name.write("\n")
    file_name.write(". ~/.bashrc\n")
    file_name.write("\n")
    file_name.write("JobName=SP"+script_name+"\n")
    file_name.write("\n")
    file_name.write("module purge\n")
    file_name.write("module load intel-compilers/2018.1.163\n")
    file_name.write("module load intel-mkl/2018.1.163\n")
    file_name.write("module load intel-mpi/2018.1.163\n")
    file_name.write("source /local/software/intel/18.0.1/"\
                    "compilers_and_libraries_2018.1.163/linux/bin/compilervars.sh intel64\n")
    file_name.write("source /local/software/intel-mkl/2018.1.163/"\
                    "compilers_and_libraries_2018.1.163/linux/mkl/bin/mklvars.sh intel64\n")
    file_name.write("export MKL_NUM_THREADS=1\n")
    file_name.write("NUMEXPR_NUM_THREADS=1\n")
    file_name.write("OMP_NUM_THREADS=1\n")
    file_name.write("((NUM_WORKER_TASKS=SLURM_NTASKS - 2))\n")
    file_name.write("\n")
    file_name.write("# Function to check if the VASP-SP calculation is properly running\n")
    file_name.write("scf_checks () {\n")
    file_name.write("TOTEN=$1\n")
    file_name.write("echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - "\
                    "Checking SCF's dE values within OSZICAR\" >> $WD/${JobName}.log\n")
    file_name.write("while [ -z \"$TOTEN\" ];do\n")
    file_name.write("  dE=$(grep 'DAV' OSZICAR | tail --line=1 | awk '{print $4}')\n")
    file_name.write("  dE2F=$(echo \"$(awk \"BEGIN {print $dE}\")\" | awk '{printf \"%.6f\", $0}')\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - SCF... dE = $dE2F\" >> $WD/${JobName}.log\n")
    file_name.write("  if [ $(echo \"$dE2F < 0\" | bc -l) -eq 1 ]; then\n")
    file_name.write("    ENER=$(echo \"$dE2F * -1\" | bc -l)\n")
    file_name.write("    if (( $(echo \"$ENER > 10000000\" | bc -l) ));then\n")
    file_name.write("      echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - "\
                    "SCF is not converging well! Stopping VASP-SP calculation.\" >> $WD/${JobName}.log\n")
    file_name.write("      vasp_sp_process_id=$(squeue --name=\"$JobName\" -h -o %A)\n")
    file_name.write("      vasopt_process_id=\"$(squeue --name=\"VASOPT"+str(molecule_name)+"\" -h -o %A)\"\n")
    file_name.write("      if [ -f \"$WD/../failed_set_of_vasp_sp.dat\" ];then\n")
    file_name.write("        HAVEIT=$(grep \"${JobName:2}\" $WD/../failed_set_of_vasp_sp.dat)\n")
    file_name.write("        if [ -z \"$HAVEIT\"  ];then\n")
    file_name.write("          echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_sp.dat\n")
    file_name.write("        fi\n")
    file_name.write("      else\n")
    file_name.write("        echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_sp.dat\n")
    file_name.write("      fi\n")
    file_name.write("      echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - Stopping VASOPT"+str(molecule_name)+".sh as well!\" >> $WD/${JobName}.log\n")
    file_name.write("      cp ${SCHED}/OSZICAR $WD/OSZICAR.failed\n")
    file_name.write("      scancel \"$vasopt_process_id\"\n")
    file_name.write("      sleep 0.5m\n")
    file_name.write("      break\n")
    file_name.write("    else\n")
    file_name.write("      TOTEN=$(grep 'free  energy   TOTEN  =' OUTCAR)\n")
    file_name.write("    fi\n")
    file_name.write("  else\n")
    file_name.write("    if (( $(echo \"$dE2F > 10000000\" | bc -l) ));then\n")
    file_name.write("      echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - "\
                    "SCF is not converging well! Stopping VASP-SP calculation.\" >> $WD/${JobName}.log\n")
    file_name.write("      vasp_sp_process_id=$(squeue --name=\"$JobName\" -h -o %A)\n")
    file_name.write("      vasopt_process_id=\"$(squeue --name=\"VASOPT"+str(molecule_name)+"\" -h -o %A)\"\n")
    file_name.write("      vasp_cancel_ids=($vasopt_process_id $vasp_sp_process_id)\n")
    file_name.write("      if [ -f \"$WD/../failed_set_of_vasp_sp.dat\" ];then\n")
    file_name.write("        HAVEIT=$(grep \"${JobName:2}\" $WD/../failed_set_of_vasp_sp.dat)\n")
    file_name.write("        if [ -z \"$HAVEIT\"  ];then\n")
    file_name.write("          echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_sp.dat\n")
    file_name.write("        fi\n")
    file_name.write("      else\n")
    file_name.write("        echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_sp.dat\n")
    file_name.write("      fi\n")
    file_name.write("#      echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_sp.dat\n")
    file_name.write("      echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - Stopping VASOPT"+str(molecule_name)+".sh as well!\" >> $WD/${JobName}.log\n")
    file_name.write("      cp ${SCHED}/OSZICAR $WD/OSZICAR.failed\n")
    file_name.write("      scancel \"$vasopt_process_id\"\n")
    file_name.write("      sleep 0.5m\n")
    file_name.write("      break\n")
    file_name.write("    else\n")
    file_name.write("      TOTEN=$(grep 'free  energy   TOTEN  =' OUTCAR)\n")
    file_name.write("    fi\n")
    file_name.write("  fi\n")
    file_name.write("  sleep 2m\n")
    file_name.write("done\n")
    file_name.write("if [ ! -z \"$vasp_sp_process_id\" ];then\n")
    file_name.write("  echo \"$vasp_sp_process_id\"\n")
    file_name.write("else\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - Check finished. SCF done!\" >> $WD/${JobName}.log\n")
    file_name.write("fi\n")
    file_name.write("}\n")
    file_name.write("\n")
    file_name.write("final_check () {\n")

    file_name.write("if [ -f \"$WD/../failed_set_of_vasp_sp.dat\" ];then\n")
    file_name.write("  FAILED=$(wc -l $WD/../failed_set_of_vasp_sp.dat | awk '{print $1}')\n")
    file_name.write("else\n")
    file_name.write("  FAILED=0\n")
    file_name.write("fi\n")
    file_name.write("\n")
    file_name.write("if [ -f \"$WD/../symmetry_fails_sp.dat\" ];then\n")
    file_name.write("  SYMMETRY_FAILS=$(wc -l $WD/../symmetry_fails_sp.dat | awk '{print $1}')\n")
    file_name.write("else\n")
    file_name.write("  SYMMETRY_FAILS=0\n")
    file_name.write("fi\n")
    file_name.write("\n")
    file_name.write("FINISHED=$(wc -l $WD/../finished.dat | awk '{print $1}')\n")
    file_name.write("ALL=$(wc -l $WD/../../Unique-"+str(vasp_sp_offset)+"/input-for-vasp-sp.txt | awk '{print $1}')\n")
    file_name.write("TOTAL=$(($FAILED+$FINISHED+${SYMMETRY_FAILS}))\n")
    file_name.write("QSTAT=$(squeue --format=\"%.10i %.9P %.14j %.8u %.8T %.10M %.10l %.3D %R\" --me | grep 'SP' )\n")
    file_name.write("NQSTAT=$(squeue --format=\"%.10i %.9P %.14j %.8u %.8T %.10M %.10l %.3D %R\" --me | grep \"SP\" | awk '{print $3}' | wc -l)\n")
    file_name.write("LASTQSTAT=$(squeue --format=\"%.10i %.9P %.14j %.8u %.8T %.10M %.10l %.3D %R\" --me | grep \"SP\" | awk '{print $3}' | tail --line=1)\n")
    file_name.write("VASOPT=$(squeue --format=\"%.10i %.9P %.14j %.8u %.8T %.10M %.10l %.3D %R\" --me | grep 'VASOPT"+str(molecule_name)+"' )\n")
    file_name.write("if [ -z \"$QSTAT\" ];then\n")
    file_name.write("  if [ $TOTAL -lt $ALL ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"SP${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - No active VASP-SP jobs, but the VASP-SP set is not complete. Submitting VASIC"+str(molecule_name)+".sh again\"\n") 
    file_name.write("    sbatch VASIC"+str(molecule_name)+".sh\n")
    file_name.write("  elif [ $TOTAL -eq $ALL ] && [ -z $VASOPT ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"SP${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - The VASP-SP set is complete. Submitting VASOPT"+str(molecule_name)+".sh\"\n")
    file_name.write("    sbatch VASOPT"+str(molecule_name)+".sh\n")
    file_name.write("  fi\n")
    file_name.write("elif [ ! -z \"$QSTAT\" ] && [ $NQSTAT -eq 1 ] && [ $LASTQSTAT == $JobName ];then\n")
    file_name.write("  if [ $TOTAL -lt $ALL ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"SP${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - No active VASP-SP jobs, but the VASP-SP set is not complete. Submitting VASIC"+str(molecule_name)+".sh again\"\n") 
    file_name.write("    sbatch VASIC"+str(molecule_name)+".sh\n")
    file_name.write("  elif [ $TOTAL -eq $ALL ] && [ -z $VASOPT ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"SP${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - The VASP-SP set is complete. Submitting VASOPT"+str(molecule_name)+".sh\"\n")
    file_name.write("    sbatch VASOPT"+str(molecule_name)+".sh\n")
    file_name.write("  fi\n")
    file_name.write("else\n")
    file_name.write("  echo \"SP${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - There are some SP#.sh active runs!\"\n")
    file_name.write("fi\n")
    file_name.write("}\n")
    file_name.write("\n")
    file_name.write("WD=`pwd`\n")
    file_name.write("cd $WD\n")
    file_name.write("\n")
    file_name.write("LOGFILE=$WD/output.VaspT-$SLURM_JOB_ID.out\n")
    file_name.write("SCHED=/scratch/rcdb1c20/calculations-dir/$JobName.$SLURM_JOB_ID\n")
    file_name.write("mkdir -p ${SCHED}\n")
    file_name.write("echo \"Scheduler file @ ${SCHED}\"\n")
    file_name.write("INPUT_FILES=\"INCAR KPOINTS POTCAR POSCAR\"\n")
    file_name.write("\n")
    file_name.write("cp -p ${INPUT_FILES} ${SCHED}; cd ${SCHED}\n")
    file_name.write("prog=~/VASP-code/vasp.5.4.4.pl2/bin/vasp_std\n")
    file_name.write("mpirun -np $SLURM_NTASKS $prog  > VaspT.out &\n")
    file_name.write("\n")
    file_name.write("rm -rf $WD/${JobName}.log\n")
    file_name.write("while [ ! -f \"OSZICAR\" ];do\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - OSZICAR not found!\" >> $WD/${JobName}.log\n")
    file_name.write("  sleep 0.5m\n")
    file_name.write("done\n")
    file_name.write("echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - OSZICAR found!\" >> $WD/${JobName}.log\n")
    file_name.write("SYM_CHECK=$(grep 'group associated with its full' OUTCAR)\n")
    file_name.write("if [ ! -z \"${SYM_CHECK}\" ];then\n")
    file_name.write("  echo ${JobName:2} >> ${WD}/../symmetry_fails_sp.dat\n")
    file_name.write("  vasopt_process_id=\"$(squeue --name=\"VASOPT"+str(molecule_name)+"\" -h -o %A)\"\n")
    file_name.write("  scancel \"$vasopt_process_id\"\n")
    file_name.write("  final_check\n")
    file_name.write("  exit\n")
    file_name.write("fi\n")
    file_name.write("DAV=$(grep 'DAV' OSZICAR)\n")
    file_name.write("while [ -z \"$DAV\" ];do\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - DAV not found!\" >> $WD/${JobName}.log\n")   
    file_name.write("  DAV=$(grep 'DAV' OSZICAR)\n")
    file_name.write("  sleep 1m\n")
    file_name.write("done\n")
    file_name.write("echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - DAV found!\" >> $WD/${JobName}.log\n")   
    file_name.write("TOTEN=$(grep 'free  energy   TOTEN  =' OUTCAR)\n")
    file_name.write("process_id=$(scf_checks \"$TOTEN\")\n")
    file_name.write("\n")
    file_name.write("if [ -z \"$process_id\" ];then\n")
    file_name.write("  cp -r OUTCAR OSZICAR CONTCAR  ${WD}\n")
    file_name.write("  cd ${WD}\n")
    file_name.write("  GOTIT=$(grep ${JobName:2} ../finished.dat)\n")
    file_name.write("  if [ -z $GOTIT ];then\n")
    file_name.write("    echo ${JobName:2} >> ../finished.dat\n")
    file_name.write("  fi\n")
    file_name.write("else\n")
    file_name.write("  cd ${WD}\n")
    file_name.write("fi\n")
    file_name.write("\n")
    file_name.write("final_check\n")
    file_name.write("\n")
    file_name.write("rm -rf ${SCHED}\n")
    file_name.write("\n")

#-- Closing file 
    file_name.close()

def VASP_opt_script_generator(script_name,partition,vasp_opt_offset,molecule_name):

#-- Opening file 
    file_name=open("QN"+script_name+".sh", "w+")

    file_name.write("#!/bin/bash\n")
    file_name.write("#SBATCH --job-name  QN"+script_name+"\n")
    file_name.write("#SBATCH --time=30:00:00\n")
    file_name.write("#SBATCH --nodes=2\n")
    file_name.write("#SBATCH --partition="+partition+"\n")
    file_name.write("#SBATCH --ntasks-per-node=40\n")
    file_name.write("#SBATCH --ntasks=80\n")
    file_name.write("\n")
    file_name.write(". ~/.bashrc\n")
    file_name.write("\n")
    file_name.write("JobName=QN"+script_name+"\n")
    file_name.write("\n")
    file_name.write("module purge\n")
    file_name.write("module load intel-compilers/2018.1.163\n")
    file_name.write("module load intel-mkl/2018.1.163\n")
    file_name.write("module load intel-mpi/2018.1.163\n")
    file_name.write("source /local/software/intel/18.0.1/"\
                    "compilers_and_libraries_2018.1.163/linux/bin/compilervars.sh intel64\n")
    file_name.write("source /local/software/intel-mkl/2018.1.163/"\
                    "compilers_and_libraries_2018.1.163/linux/mkl/bin/mklvars.sh intel64\n")
    file_name.write("export MKL_NUM_THREADS=1\n")
    file_name.write("NUMEXPR_NUM_THREADS=1\n")
    file_name.write("OMP_NUM_THREADS=1\n")
    file_name.write("((NUM_WORKER_TASKS=SLURM_NTASKS - 2))\n")
    file_name.write("\n")
    file_name.write("backing_up_output_files () {\n")
    file_name.write("echo -e \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - Finished ENCUT=$1 run and backing up files to the WD: \"\n")
    file_name.write("cp INCAR INCAR.$1; echo \"~          + INCAR copied to INCAR.$1\"\n")
    file_name.write("mv OUTCAR OUTCAR.$1; echo \"~          + OUTCAR moved to OUTCAR.$1\"\n")
    file_name.write("mv OSZICAR OSZICAR.$1; echo \"~          + OSZICAR moved to OSZICAR.$1\"\n")
    file_name.write("cp CONTCAR CONTCAR.$1; echo \"~          + CONTCAR copied to CONTCAR.$1\"\n")
    file_name.write("cp CONTCAR POSCAR; echo \"~          + CONTCAR copied to POSCAR\"\n")
    file_name.write("if [ $1 -eq \"300\" ];then\n")
    file_name.write("  cp -r OUTCAR.$1 OSZICAR.$1 CONTCAR.$1 INCAR.$1 ${WD}\n")
    file_name.write("  cp -r WAVECAR ${WD}/WAVECAR.$1\n")
    file_name.write("  echo \"~          + WAVECAR copied to WAVECAR.$1\"\n")
    file_name.write("else\n")
    file_name.write("  cp -r OUTCAR.$1 ${WD}\n")
    file_name.write("  cp -r OSZICAR.$1 ${WD}\n")
    file_name.write("  cp -r CONTCAR.$1 ${WD}\n")
    file_name.write("  cp -r INCAR.$1 ${WD}\n")
    file_name.write("  rm -rf $WD/WAVECAR.*\n")
    file_name.write("fi\n")
    file_name.write("}\n")
    file_name.write("\n")
    file_name.write("# Function to check if the VASP optimisation is properly running\n")
    file_name.write("scf_checks () {\n")
    file_name.write("ENER_IN=$1\n")
    file_name.write("GEOM=$2\n")
    file_name.write("echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - Checking SCF's dE values within OSZICAR\" >> $WD/${JobName}.log\n")
    file_name.write("while [ -z \"$GEOM\" ];do\n")
    file_name.write("  QNSTEP=$(grep 'Iteration' OUTCAR | tail -n 1 | awk '{print $3}')\n")
    file_name.write("  dE=$(grep 'DAV' OSZICAR | tail --line=1 | awk '{print $4}')\n")
    file_name.write("  dE2F=$(echo \"$(awk \"BEGIN {print $dE}\")\" | awk '{printf \"%.6f\", $0}')\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - SCF... dE = ${dE2F}. QN step ${QNSTEP:: -1}\" >> $WD/${JobName}.log\n")
    file_name.write("  if [ $(echo \"$dE2F < 0\" | bc -l) -eq 1 ]; then\n")
    file_name.write("    ENER=$(echo \"$dE2F * -1\" | bc -l)\n")
    file_name.write("    if (( $(echo \"$ENER > 10000000\" | bc -l) ));then\n")
    file_name.write("      echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - SCF is not converging well! Stopping VASP optimisation.\" >> $WD/${JobName}.log\n")
    file_name.write("      vasp_opt_process_id=$(squeue --name=\"$JobName\" -h -o %A)\n")
    file_name.write("      if [ -f \"$WD/../failed_set_of_vasp_opt.dat\" ];then\n")
    file_name.write("        HAVEIT=$(grep \"${JobName:2}\" $WD/../failed_set_of_vasp_opt.dat)\n")
    file_name.write("        if [ -z \"$HAVEIT\"  ];then\n")
    file_name.write("          echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_opt.dat\n")
    file_name.write("        fi\n")
    file_name.write("      else\n")
    file_name.write("        echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_opt.dat\n")
    file_name.write("      fi\n")
    file_name.write("      cp ${SCHED}/OSZICAR $WD/OSZICAR.failed\n")
    file_name.write("      sleep 0.5m\n")
    file_name.write("      break\n")
    file_name.write("    else\n")
    file_name.write("      GEOM=$(grep \"reached required accuracy - stopping structural energy minimisation\" OUTCAR)\n")
    file_name.write("    fi\n")
    file_name.write("  else\n")
    file_name.write("    if (( $(echo \"$dE2F > 10000000\" | bc -l) ));then\n")
    file_name.write("      echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - SCF is not converging well! Stopping VASP optimisation.\" >> $WD/${JobName}.log\n")
    file_name.write("      vasp_opt_process_id=$(squeue --name=\"$JobName\" -h -o %A)\n")
    file_name.write("      if [ -f \"$WD/../failed_set_of_vasp_opt.dat\" ];then\n")
    file_name.write("        HAVEIT=$(grep \"${JobName:2}\" $WD/../failed_set_of_vasp_opt.dat)\n")
    file_name.write("        if [ -z \"$HAVEIT\"  ];then\n")
    file_name.write("          echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_opt.dat\n")
    file_name.write("        fi\n")
    file_name.write("      else\n")
    file_name.write("        echo \"${JobName:2}\" >> $WD/../failed_set_of_vasp_opt.dat\n")
    file_name.write("      fi\n")
    file_name.write("      cp ${SCHED}/OSZICAR $WD/OSZICAR.failed\n")
    file_name.write("      sleep 0.5m\n")
    file_name.write("      break\n")
    file_name.write("    else\n")
    file_name.write("      GEOM=$(grep \"reached required accuracy - stopping structural energy minimisation\" OUTCAR)\n")
    file_name.write("    fi\n")
    file_name.write("  fi\n")
    file_name.write("  cp CONTCAR $WD/CONTCAR.${ENER_IN}\n")
    file_name.write("  sleep 5m\n")
    file_name.write("done\n")
    file_name.write("if [ ! -z \"$vasp_opt_process_id\" ];then\n")
    file_name.write("  echo \"$vasp_opt_process_id\"\n")
    file_name.write("else\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - Check finished. SCF done!\" >> $WD/${JobName}.log\n")
    file_name.write("fi\n")
    file_name.write("}\n")
    file_name.write("\n")
    file_name.write("final_check () {\n")
    file_name.write("if [ -f \"$WD/../failed_set_of_vasp_opt.dat\" ];then\n")
    file_name.write("  FAILED=$(wc -l $WD/../failed_set_of_vasp_opt.dat | awk '{print $1}')\n")
    file_name.write("else\n")
    file_name.write("  FAILED=0\n")
    file_name.write("fi\n")
    file_name.write("\n")
    file_name.write("if [ -f \"$WD/../symmetry_fails_opt.dat\" ];then\n")
    file_name.write("  SYMMETRY_FAILS=$(wc -l $WD/../symmetry_fails_opt.dat | awk '{print $1}')\n")
    file_name.write("else\n")
    file_name.write("  SYMMETRY_FAILS=0\n")
    file_name.write("fi\n")
    file_name.write("\n")
    file_name.write("if [ -f \"$WD/../finished.dat\" ];then\n")
    file_name.write("  FINISHED=$(wc -l $WD/../finished.dat | awk '{print $1}')\n")
    file_name.write("else\n")
    file_name.write("  FINISHED=0\n")
    file_name.write("fi\n")
    file_name.write("\n")
    file_name.write("ALL=$(wc -l $WD/../input-for-vasp-opt.txt | awk '{print $1}')\n")
    file_name.write("TOTAL=$(($FAILED+$FINISHED+${SYMMETRY_FAILS}))\n")
    file_name.write("QSTAT=$(squeue --format=\"%.10i %.9P %.14j %.8u %.8T %.10M %.10l %.3D %R\" --me | grep 'QN' )\n")
    file_name.write("NQSTAT=$(squeue --format=\"%.10i %.9P %.14j %.8u %.8T %.10M %.10l %.3D %R\" --me | grep \"QN\" | awk '{print $3}' | wc -l)\n")
    file_name.write("LASTQSTAT=$(squeue --format=\"%.10i %.9P %.14j %.8u %.8T %.10M %.10l %.3D %R\" --me | grep \"QN\" | awk '{print $3}' | tail --line=1)\n")
    file_name.write("if [ -z \"$QSTAT\" ];then\n")
    file_name.write("  if [ $TOTAL -lt $ALL ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"QN${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - No active VASP-OPT jobs, but the VASP-OPT set is not complete. Submitting VASOPT"+str(molecule_name)+".sh again\"\n") 
    file_name.write("    sbatch VASOPT"+str(molecule_name)+".sh\n")
    file_name.write("  elif [ $TOTAL -eq $ALL ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"QN${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - The VASP-OPT set is complete.\"\n")
    file_name.write("  fi\n")
    file_name.write("elif [ ! -z \"$QSTAT\" ] && [ $NQSTAT -eq 1 ] && [ $LASTQSTAT == $JobName ];then\n")
    file_name.write("  if [ $TOTAL -lt $ALL ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"QN${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - No active VASP-OPT jobs, but the VASP-OPT set is not complete. Submitting VASOPT"+str(molecule_name)+".sh again\"\n") 
    file_name.write("    sbatch VASOPT"+str(molecule_name)+".sh\n")
    file_name.write("  elif [ $TOTAL -eq $ALL ];then\n")
    file_name.write("    cd $WD/../..\n")
    file_name.write("    echo \"QN${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - The VASP-OPT set is complete.\"\n")
    file_name.write("  fi\n")
    file_name.write("else\n")
    file_name.write("  echo \"QN${var}:    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - There are some QN#.sh active runs!\"\n")
    file_name.write("fi\n")
    file_name.write("}\n")
    file_name.write("\n")
    file_name.write("WD=`pwd`\n")
    file_name.write("cd $WD\n")
    file_name.write("\n")
    file_name.write("LOGFILE=$WD/output.VaspT-$SLURM_JOB_ID.out\n")
    file_name.write("SCHED=/scratch/rcdb1c20/calculations-dir/$JobName.$SLURM_JOB_ID\n")
    file_name.write("mkdir -p ${SCHED}\n")
    file_name.write("echo \"Scheduler file @ ${SCHED}\"\n")
    file_name.write("INPUT_FILES=\"INCAR KPOINTS POTCAR POSCAR\"\n")
    file_name.write("\n")
    file_name.write("cp -p ${INPUT_FILES} ${SCHED}; cd ${SCHED}\n")
    file_name.write("prog=~/VASP-code/vasp.5.4.4.pl2/bin/vasp_std\n")
    file_name.write("\n")
    file_name.write("OLD_ISTART=\"ISTART = 0\"\n")
    file_name.write("OLD_ENCUT=\"ENCUT = 300\"\n")
    file_name.write("ENCUT=(\"300\" \"500\")\n")
    file_name.write("rm -rf $WD/${JobName}.log\n")
    file_name.write("for ENER in \"${ENCUT[@]}\"; do\n")
    file_name.write("\n")
    file_name.write("  NEW_ENCUT=\"ENCUT = $ENER\"\n")
    file_name.write("  sed -i \"s%${OLD_ENCUT}%${NEW_ENCUT}%g\" ./INCAR\n")
    file_name.write("\n")
    file_name.write("  mpirun -np $SLURM_NTASKS $prog  > VaspT.out &\n")
    file_name.write("\n")
    file_name.write("  while [ ! -f \"OSZICAR\" ];do\n")
    file_name.write("    echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - OSZICAR not found!\" >> $WD/${JobName}.log\n")
    file_name.write("    sleep 0.5m\n")
    file_name.write("  done\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - OSZICAR found!\" >> $WD/${JobName}.log\n")
    file_name.write("  SYM_CHECK=$(grep 'group associated with its full' OUTCAR)\n")
    file_name.write("  if [ ! -z \"${SYM_CHECK}\" ];then\n")
    file_name.write("    echo ${JobName:2} >> ${WD}/../symmetry_fails_opt.dat\n")
    file_name.write("    final_check\n")
    file_name.write("    exit\n")
    file_name.write("  fi\n")
    file_name.write("  DAV=$(grep 'DAV' OSZICAR)\n")
    file_name.write("  while [ -z \"$DAV\" ];do\n")
    file_name.write("    echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - DAV not found!\" >> $WD/${JobName}.log\n")
    file_name.write("    DAV=$(grep 'DAV' OSZICAR)\n")
    file_name.write("    sleep 2m\n")
    file_name.write("  done\n")
    file_name.write("  echo \"~    + [$(date +%Y-%m-%d)] $(date +%H:%M:%S) - DAV found!\" >> $WD/${JobName}.log\n")
    file_name.write("  GEOM=$(grep \"reached required accuracy - stopping structural energy minimisation\" OUTCAR)\n")
    file_name.write("\n")
    file_name.write("  process_id=$(scf_checks \"$ENER\" \"$GEOM\")\n")
    file_name.write("\n")
    file_name.write("  if [ -z \"$process_id\" ];then\n")
    file_name.write("    backing_up_output_files $ENER\n")
    file_name.write("    if [ -f \"${WD}/../finished.dat\" ];then\n")
    file_name.write("      GOTIT=$(grep ${JobName:2} ${WD}/../finished.dat)\n")
    file_name.write("    else\n")
    file_name.write("      GOTIT=\n")
    file_name.write("    fi\n")
    file_name.write("    NEW_ISTART=\"ISTART = 1\"\n")
    file_name.write("    sed -i \"s%${OLD_ISTART}%${NEW_ISTART}%g\" ./INCAR\n")
    file_name.write("    if [ -z $GOTIT ] && [ $ENER -eq \"500\" ];then\n")
    file_name.write("      echo ${JobName:2} >> ${WD}/../finished.dat\n")
    file_name.write("      cd $WD\n")
    file_name.write("      rm -rf ${SCHED}\n")
    file_name.write("      break\n")
    file_name.write("    fi\n")
    file_name.write("  else\n")
    file_name.write("    cd ${WD}\n")
    file_name.write("    break\n")
    file_name.write("  fi\n")
    file_name.write("\n")
    file_name.write("#  [ $ENER -eq 500 ] && break\n")
    file_name.write("\n")
    file_name.write("done\n")
    file_name.write("\n")
    file_name.write("final_check\n")


#-- Closing file 
    file_name.close()

class FMCSP_wf:

    def __init__(self):

#------ I will need the user_name and conda_env for the generation of the slurm scripts afterwards
        self.slurm_script_generation_options = {}
        self.slurm_script_generation_options["user_name"] = \
                   subprocess.check_output("whoami",shell=True,text=True).rstrip() 
        self.slurm_script_generation_options["conda_environment"] = \
                   subprocess.check_output("echo $CONDA_DEFAULT_ENV",shell=True,text=True).rstrip()

#------ Reading the fmcspwf toml file 
        if not os.path.exists("fmcspwf.toml"):
            raise ValueError(
                 "File fmcspwf.toml must be within the working directory!!"
            )
        else:
            with open('fmcspwf.toml', 'r') as fmcspwf_file:
                self.fmcsp_data = toml.load(fmcspwf_file)

#-------------- Re-organising self.fmcsp_data as a simple dictionary: {'key': value, ...} 
                for section, values in self.fmcsp_data.items():
                    for key, value in values.items():
                        self.slurm_script_generation_options[key] = value

        self.saved_nodes = self.slurm_script_generation_options["number_of_nodes"] 
        self.saved_conda = self.slurm_script_generation_options["conda_environment"] 

#------ Number of space groups and valid crystals  
        if self.fmcsp_data['fmcsp']['space_group_list'] in COMMON_SAMPLING_SETTINGS:
            self.sg = COMMON_SAMPLING_SETTINGS[self.fmcsp_data['fmcsp']['space_group_list']]
            self.structures = { x: self.sg["number_structures"][x] for x in self.sg["space_group"] }
        else:
            if self.fmcsp_data['fmcsp']['n_crystals'] is None:
                raise ValueError(
                    "Must set number of structures for custom spacegroup set"
                )
            try:
                self.structures = { int(x): self.fmcsp_data['fmcsp']['n_crystals'] 
                                    for x in self.fmcsp_data['fmcsp']['space_group_list'].split(",") }
            except ValueError as e:
                LOG.error("Error interpreting requested spacegroups: %s", e)
                raise ValueError("Invalid spacegroup") from e
        self.slurm_script_generation_options["number_of_space_groups"] = len(self.structures)
        self.slurm_script_generation_options["structures"] = self.structures
        self.slurm_script_generation_options["space_group_list"] = list(self.structures)

        self.slurm_script_generation_options["number_of_torsions"] = \
                  dict(Counter(self.slurm_script_generation_options["torsions_atoms_list"]))

#------ The file to save the processes ids of all the submitted jobs
        self.processes_ids_file=open("processes_ids.txt", "w+")

    def cdb_step(self):

        self.slurm_script_generation_options["script_name"]="CDB"
        with open("molecules_list.txt", 'r') as f:

#---------- SLURM CDB scripts generation and submission
            nmol=1
            mol_ids={}
            print("main::    ")
            print("main::   - Submitting CDB calculations")
            print("main::    ")
            for line in f:
                xyz_file_name=line.split()[0]
                cdb = FMCSPScriptsClass(self.slurm_script_generation_options)
                cdb.slurm_script_for_fmcsp_generator("",xyz_file_name[:-4])
                mol_ids[nmol]=cdb.slurm_script_submission(xyz_file_name[:-4])
                self.processes_ids_file.write(mol_ids[nmol]+"\n")
                nmol += 1
            print("main::    ")
            print("main::        - The conformational DB IDs dictionary is: ", mol_ids)

#------ Joining all the CDBs in case there are more than 1, otherwise its name is modified 
        main_name=self.slurm_script_generation_options["molecule_name"]
        self.slurm_script_generation_options["script_name"]="jointDB"
        self.slurm_script_generation_options["number_of_nodes"]=1
        joint_cdbs = FMCSPScriptsClass(self.slurm_script_generation_options)
        joint_cdbs.slurm_script_for_fmcsp_generator("",str(main_name))
        list_of_mol_ids = ':'.join(list(mol_ids.values()))
        jointdb_submission="sbatch --dependency=afterok:"+list_of_mol_ids+" jointDB"+str(main_name)+".sh"
        print("main::        - Joining the computed conformational DBs: ", jointdb_submission)
        process_id=subprocess.run(jointdb_submission, capture_output=True, text=True, shell=True)
        jointdb_id=process_id.stdout.strip().split()[3]
        self.processes_ids_file.write(jointdb_id+"\n")
        print("main::        - The jointDB"+str(main_name)+"'s ID is: ", jointdb_id)
        return jointdb_id

    def fmcsp_step(self,jointdb_id):

#------ Do we have what we need to re-start the workflow at FMCSP step?
        main_name=self.slurm_script_generation_options["molecule_name"]
        if jointdb_id == "":
            if not os.path.exists(str(main_name)+"_flex.db"):
                message=str(main_name)+"_flex.db does not exist! Please, generate it using mol-dis."
                print(message)
                exit(message)

#------ SLURM FMCSP scripts generation and submission
        self.slurm_script_generation_options["script_name"]="FMCSP"
        self.slurm_script_generation_options["number_of_nodes"]=self.saved_nodes
        print("main::    ")
        print("main::   - Submitting FMCSP calculations")
        print("main::    ")
        sg_ids={}
        for space_group in list(self.slurm_script_generation_options["structures"].keys()):
            fmcsp = FMCSPScriptsClass(self.slurm_script_generation_options)
            fmcsp.slurm_script_for_fmcsp_generator(space_group,str(main_name))
            if jointdb_id == "":
                fmcsp_submission="sbatch FMCSP"+str(main_name)+"_"+str(space_group)+".sh"
            else:
                fmcsp_submission="sbatch --dependency=afterok:"+str(jointdb_id)+" FMCSP"+str(main_name)+"_"+str(space_group)+".sh"
            print("main::        + Submitting FMCSP"+str(main_name)+str(space_group)+" calculations: ", fmcsp_submission)
            process_id=subprocess.run(fmcsp_submission, capture_output=True, text=True, shell=True)
            sg_ids[str(space_group)]=process_id.stdout.strip().split()[3]
            self.processes_ids_file.write(sg_ids[str(space_group)]+"\n")
        print("main::    ")
        print("main::        - The FMCSP IDs dictionary is: ", sg_ids)
 
        if len(self.slurm_script_generation_options["structures"]) > 1:
            self.slurm_script_generation_options["script_name"]="jointSG"
            joint_sg = FMCSPScriptsClass(self.slurm_script_generation_options)
            joint_sg.slurm_script_for_fmcsp_generator("",str(main_name))
            list_of_sg_ids = ':'.join(list(sg_ids.values()))
            jointsg_submission="sbatch --dependency=afterok:"+list_of_sg_ids+" jointSG"+str(main_name)+".sh"
            print("main::        - Joining the computed SGs' DBs: ", jointsg_submission)
            process_id=subprocess.run(jointsg_submission, capture_output=True, text=True, shell=True)
            jointsg_id=process_id.stdout.strip().split()[3]
            print("main::        - The jointSG"+str(main_name)+"'s ID is : ", jointsg_id)
        if len(self.slurm_script_generation_options["structures"]) == 1:
            jointsg_id=sg_ids[str(space_group)]
        self.processes_ids_file.write(jointsg_id+"\n")
        return jointsg_id

    def dftb_step(self,jointsg_id):

#------ Do we have what we need to re-start the workflow at DFTB step?
        main_name=self.slurm_script_generation_options["molecule_name"]
        if jointsg_id == "":

            dftb_offset=self.slurm_script_generation_options["dftb_offset_energy"]
            if len(self.slurm_script_generation_options["structures"]) > 1:

#-------------- Do we have all-SGs.db and structures.csv to re-start the workflow at DFTB step?
                message="all-SGs.db or structures.csv does not exist! Please, generate them running FMCSP step."
                if not os.path.exists("all-SGs.db") and not os.path.exists("structures.csv"):
                    print(message)
                    exit(message)

#-------------- Do we have main_name-(SG).db and structures-(SG).csv to re-start the workflow at DFTB step?
                for sg in list(self.slurm_script_generation_options["structures"]):
                    message=str(main_name)+"-"+str(sg)+".db or structures-"+str(sg)+ \
                            ".csv does not exist! Please, generate them running FMCSP step."
                    if not os.path.exists(str(main_name)+"-"+str(sg)+".db") and \
                        not os.path.exists("structures-"+str(sg)+".csv"):
                        print(message)
                        exit(message)
 
                    message="dftb_input_"+str(sg)+"_"+str(dftb_offset)+".db input file " \
                            " does not exist! Please, generate it running jointSG.sh before run DFTB step."
                    if not os.path.exists("dftb_input_"+str(sg)+"_"+str(dftb_offset)+".db"):
                        print(message)
                        exit(message)
 
            else:
                space_group=self.slurm_script_generation_options["space_group_list"]
                message=str(main_name)+"-"+space_group+".db or structures-"+space_group+ \
                        ".csv does not exist! Please, generate them running FMCSP step."
                if not os.path.exists(str(main_name)+"-"+space_group+".db") and \
                   not os.path.exists("structures-"+space_group+".csv") and \
                   not os.path.exists("dftb_input_"+space_group+"_"+str(dftb_offset)+".db"):
                    print(message)
                    exit(message)

#------ SLURM DFTB scripts and input db generation 
        self.slurm_script_generation_options["script_name"]="DFTB"
        self.slurm_script_generation_options["number_of_nodes"]=self.saved_nodes
        print("main::    ")
        print("main::   - Submitting DFTB re-optimisations")
        print("main::    ")
        dftb = FMCSPScriptsClass(self.slurm_script_generation_options)
        dftb_ids={}
        for sg in list(self.slurm_script_generation_options["structures"].keys()):
            dftb.slurm_script_for_fmcsp_generator(sg,str(main_name))
            if jointsg_id == "":
                dftb_submission="sbatch DFTB"+str(main_name)+"_"+str(sg)+".sh"
            else:
                dftb_submission="sbatch --dependency=afterok:"+str(jointsg_id)+" DFTB"+str(main_name)+"_"+str(sg)+".sh"
            print("main::        + Submitting DFTB re-optimisations: ", dftb_submission)
            process_id = subprocess.run(dftb_submission, capture_output=True, text=True, shell=True)
            dftb_ids[str(sg)] = process_id.stdout.strip().split()[3]
            self.processes_ids_file.write(dftb_ids[str(sg)]+"\n")
        print("main::    ")
        print("main::        - The DFTB IDs dictionary is: ", dftb_ids)
        return dftb_ids

    def cmpck_step(self,dftb_id):

#------ SLURM CMPCK script generation and submission 
        main_name=self.slurm_script_generation_options["molecule_name"]
        self.slurm_script_generation_options["script_name"]="CMPCK"
        print("main::    ")
        self.slurm_script_generation_options["conda_environment"]="flex+dftb"
        self.slurm_script_generation_options["number_of_nodes"]=1
        cmpck_ids={}
        cmpck = FMCSPScriptsClass(self.slurm_script_generation_options)
        for sg in list(self.slurm_script_generation_options["structures"].keys()):
            cmpck.slurm_script_for_fmcsp_generator(sg,str(main_name))
            if dftb_id == "":
                cmpck_submission="sbatch CMPCK"+str(main_name)+"_"+str(sg)+".sh"
            else:
                cmpck_submission="sbatch --dependency=afterok:"+str(dftb_id[str(sg)])+" CMPCK"+str(main_name)+"_"+str(sg)+".sh"
            print("main::        + Submitting CMPCK DFTB clustering: ", cmpck_submission)
            process_id = subprocess.run(cmpck_submission, capture_output=True, text=True, shell=True)
            cmpck_ids[str(sg)] = process_id.stdout.strip().split()[3]
            self.processes_ids_file.write(cmpck_ids[str(sg)]+"\n")
        print("main::    ")
        print("main::        - The CMPCK IDs dictionary is: ", cmpck_ids)
        return cmpck_ids

    def vasp_sp_step(self,cmpck_id):

#------ Do we have what we need to re-start the workflow at VASP-SP step?
        if cmpck_id == "":
            vasp_sp_offset=self.slurm_script_generation_options["vasp_sp_offset_energy"]
            if not os.path.exists("./Unique-"+str(vasp_sp_offset)+"/input-for-vasp-sp.txt"):
                message="Unique-"+str(vasp_sp_offset)+\
                        "/input-for-vasp-sp.txt does not exist! Please, generate it using DFTB step!"
                print(message)
                exit(message)

#------ SLURM VASIC{molecule_name}.sh script generation
        main_name=self.slurm_script_generation_options["molecule_name"]
        self.slurm_script_generation_options["script_name"]="VASP-SP"
        self.slurm_script_generation_options["conda_environment"]=self.saved_conda
        print("main::    ")
        print("main::   - Submitting VASP-SP calculations script")
        print("main::    ")
        vaspscript = FMCSPScriptsClass(self.slurm_script_generation_options)
        vaspscript.VASIC_scripts_generator()
        if cmpck_id ==- "":
            vaspscript_submission="sbatch VASIC"+str(main_name)+".sh"
        else:
            list_of_cmpck_ids = ':'.join(list(cmpck_id.values()))
            vaspscript_submission="sbatch --dependency=afterok:"+list_of_cmpck_ids+" VASIC"+str(main_name)+".sh"
        print("main::        + Submitting VASIC: ", vaspscript_submission)
        process_id = subprocess.run(vaspscript_submission, capture_output=True, text=True, shell=True)
        vasic_id = process_id.stdout.strip().split()[3]
        self.processes_ids_file.write(vasic_id+"\n")
        print("main::    ")
        print("main::        - The VASIC"+str(main_name)+".sh ID is: ", vasic_id)

    def vasp_opt_step(self,run=False):

#------ SLURM VASOPT{molecule_name}.sh script generation
        print("main::    ")
        print("main::   - Submitting VASP-OPT calculations script")
        main_name=self.slurm_script_generation_options["molecule_name"]
        vaspoptscript = FMCSPScriptsClass(self.slurm_script_generation_options)
        vaspoptscript.VASOPT_scripts_generator()
        vaspoptscript.VASOPT_check_script_generator()
        if run:
            vaspoptscript_submission="sbatch VASOPT"+str(main_name)+".sh"
            process_id = subprocess.run(vaspoptscript_submission, capture_output=True, text=True, shell=True)
            vasopt_id = process_id.stdout.strip().split()[3]
            self.processes_ids_file.write(vasopt_id+"\n")
            print("main::    ")
            print("main::        + Building VASOPT"+str(main_name)+".sh script and run it")
            print("main::    ")
            print("main::        - The VASOPT"+str(main_name)+".sh ID is: ", vasopt_id)
        else:
            print("main::    ")
            print("main::        + Building VASOPT"+str(main_name)+".sh script (this script will run at the end of VASIC"+str(main_name)+".sh)")
            print("main::    \n")

