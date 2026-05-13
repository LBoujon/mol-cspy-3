"""conf: Application for generating and clustering conformations for flexible molecules."""

__author__      = "James Bramley"
__email__   = "J.Bramley@soton.ac.uk"


# REQUIREMENTS!!!
# Installation of xtb and CREST.
# Obabel

# imports
import argparse
import logging
from cspy.configuration import CONFIG
from cspy.conf_search.mcrestfunctions import *
import os
LOG = logging.getLogger(__name__)

#Distributes cores among all CREST searches.
# print("Running with {} cores.".format(args.processors))

def get_starting_positions(input_xyz, number_of_conformers, pca_method, cluster_method, n_starting_positions, plot='False', ):
    conformers, energies = generate_RDKit_conformers(input_xyz, number_of_conformers)
    torsion_data = perform_PCA(conformers, method=pca_method)
    # LOG.info(f"There are {args.starting_positions} starting positions.")
    if not n_starting_positions:
        n_starting_positions=0
    num_starting_pos = cluster_data(torsion_data, method=cluster_method, plot=plot, k_opt=n_starting_positions)
    starting_positions = pick_from_cluster(energies, num_starting_pos, by='Energy')
    return starting_positions

def get_conformers(conf_no_list):
    clust_no=0
    xyz_name_list=[]
    for conf_no in conf_no_list:
        clust_no+=1
        bashCommand = "obabel conformers.xyz -f {} -l {} -O crest_{}.xyz".format(conf_no, conf_no, clust_no)
        output = subprocess.run(bashCommand.split(), capture_output=True)
        # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        # output, error = process.communicate()
        # print("This is the output", output.stdout)
        # print("This is the error", error)
        # LOG.info(output)
        # LOG.error(error)
        xyz_name_list.append("crest_"+str(clust_no)+".xyz")
    return xyz_name_list

def run_crest(molecule, processes):
    bashCommand = "crest {} -T {} -v4 -ewin 9".format(molecule, processes)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
def crest(molecule, torsions, processors):
    xtb(molecule, processors)
    run_crest(molecule, processors)
    cluster_searches_single(molecule, torsions)

def xtb(molecule, processes):
    bashCommand = "xtb {} -gfn 2 -opt -P {}".format(molecule, processes)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.rename("xtbopt.xyz", molecule)

def cluster_searches(molecule, torsions_file, crest_runs):
    bashCommand = "python -m cspy.conf_search.cluster_from_crest -xyz {} -t {} -mc {} -o".format(molecule, torsions_file, crest_runs)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def cluster_searches_single(molecule, torsions_file):
    bashCommand = "python -m cspy.conf_search.cluster_from_crest -xyz {} -t {} -c -o".format(molecule, torsions_file)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def cluster_using_torsions(energies_file, torsions_file, energy_window, max_rms_angle, single_angle_max, processors, overlay):
    bashCommand = "python -m cspy.conf_search.dihedral -e {} -t {} -ewin {} -rms {} -smax {} -np {}".format(energies_file, torsions_file, energy_window, max_rms_angle, single_angle_max, processors)
    if overlay == True:
        bashCommand+=" -o"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def cluster_using_rmsd(energies_file, energy_window, rmsd, processors):
    bashCommand = "python -m cspy.conf_search.rmsd -e {} -ewin {} -rmsd {} -np {}".format(energies_file, energy_window, rmsd, processors)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def mcrest(xyz, conformers, pca, cluster, plot, processors, n_starting_positions, torsions):
    LOG.info('Generating starting positions.')
    starting_positions = get_starting_positions(xyz, conformers, pca, cluster, n_starting_positions, plot)
    if plot == 'True':
        exit()
    available_processors=processors
    crest_runs=len(starting_positions)
    # processors_per_run=int(available_processors/crest_runs)
    # print(processors_per_run)
    xyz_name_list = get_conformers(starting_positions)
    for run in xyz_name_list:
        LOG.info(f"Starting CREST search for {run}.")
        dir=run.split('.xyz')[0]
        if not os.path.isdir(dir):
            os.mkdir(dir)
        os.rename(run, dir+"/"+run)
        os.chdir(dir)
        xtb(run, available_processors)
        run_crest(run, available_processors)
        os.chdir("..")
    cluster_searches(xyz, torsions, crest_runs)

def main():
    # Argparse Setup
    parser = argparse.ArgumentParser(description='Application for generating and clustering conformations for flexible molecules.')

    #General arguments
    parser.add_argument('-xyz', '--xyz_file', type=str, metavar='', help='xyz file for conformational search.')
    parser.add_argument('-t', '--torsions', type=str,default='torsions.txt', metavar='', help='List of torsion angles for the molecule equal to the first conformer of the series. #If none selected, torsional angles will be generated automatically. (to be added)')
    parser.add_argument('-np', '--processors', type=int, metavar='', default=1, help='Number of processors used.')
    parser.add_argument('--search', type=str, metavar='', default=False, help='Perform a conformational search: CREST, mCREST')
    parser.add_argument('--cluster', type=str, default=False, metavar='', help='Cluster method: TORSIONS, RMSD.')
    parser.add_argument("--log-level", default=CONFIG.get("csp.log_level"), help="Log level")

    # parser.add_argument('-dm', '--distance_metric', type=str, default='euclidean', metavar='', help='Specify distance metric used to find ideal number of starting positions.')

    #mCREST arguments
    parser.add_argument('-c', '--conformers', type=int, metavar='', default=1000, help='Number of conformers to generated to find starting positions.')
    parser.add_argument('-p','--pca', type=str, default='dpca', metavar='', help='Method used for principle component analysis: dpca, geopca')
    parser.add_argument('-dry', '--plot', action='store_true', help='Performs a dry run for the mCREST search, generates statistics for viewing data and conformer variability.')
    parser.add_argument('-mcl', '--mcluster', type=str, default='kmeans', metavar='', help='Specify clustering algorithm to find ideal number of starting positions: kmeans')
    parser.add_argument('-k', '--n_starting_positions', type=int, default=None, metavar='', help='Specifiy number of CREST searches to be carried out. If none selected, this is be assigned automatically.')
    parser.add_argument('--no-cleanup', action="store_true", help='Stops cleanup of intermediate crest searches.')

    #clustering arguments
    parser.add_argument('-e','--energies', type=str, metavar='', help='Name of energies file, holds a list of all conformers and their associated energies in kj/mol')
    parser.add_argument('-ewin', '--energy_window', type=float, metavar='', default=5, help='Energy window in which conformers should be compared in kj/mol')
    parser.add_argument('-rms', '--max_rms_angle', type=float, metavar='', default=5.0, help='Maximum rms angle between selected torsions.')
    parser.add_argument('-smax', '--single_angle_max', type=float, metavar='', default=10.0, help='Maximum single torsion angle between selected torsions.')
    parser.add_argument('-o', '--overlay', action='store_true', help='First perform an overlay between conformations.')
    parser.add_argument('-rmsd', '--max_rmsd', type=float, metavar='', default=0.5, help='Maximum rmsd between molecules for clustering.')

    args=parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format='%(asctime)s - %(levelname)s - %(module)s %(lineno)d - '
            '%(message)s'
        )

    #perform a conformer search
    if not args.search == False:
        #create torsions.txt file if one was not provided - temporary bug fix will be removed in future code improvements
        if not os.path.isfile(args.torsions):
            with open(args.torsions, mode='a'): pass
        if args.search == 'CREST':
            LOG.info("Run CREST search")
            crest(args.xyz_file, args.torsions, args.processors)
        elif args.search == 'mCREST':
            LOG.info("Run mCREST search")
            if args.plot:
                LOG.info("Performing dryrun.")
                plot='True'
            else:
                plot='False'
            mcrest(args.xyz_file, args.conformers, args.pca, args.mcluster, plot, args.processors, args.n_starting_positions, args.torsions)
        else:
            LOG.error("Did not recognise search method. Options: CREST, mCREST")

    #perform clustering
    elif not args.cluster == False:
        if args.cluster == 'TORSIONS':
            if not args.torsions:
                LOG.error("No torsions file found. Please provide torsions file.")
                exit()
            LOG.info("Clustering files uning torsions.")
            LOG.info("Using file: %s", args.energies)
            LOG.info("Using torsions in: %s", args.torsions)
            LOG.info("Energy window: %s kJ/mol", args.energy_window)
            LOG.info("Maximum RMS angle: %s%s", args.max_rms_angle, chr(176))
            LOG.info("Single angle maximum: %s%s", args.single_angle_max, chr(176))
            LOG.info("Number of processors: %s", args.processors)
            LOG.info("Overlay set to: %s", args.overlay)
            cluster_using_torsions(args.energies, args.torsions, args.energy_window, args.max_rms_angle, args.single_angle_max, args.processors, args.overlay)
            LOG.info("Clustering complete!")
        elif args.cluster == 'RMSD':
            LOG.info("Clustering files using RMSD.")
            LOG.info("RMSD threshold: %s Angstroms", args.max_rmsd)
            if args.energies is None:
                LOG.error('Files for clustering not defined. Please provide energies file with --energies.')
                exit()
            try:
                cluster_using_rmsd(args.energies, args.energy_window, args.max_rmsd, args.processors)
            except:
                print("I error")
            LOG.info("Clustering complete!")
        # elif args.cluster == 'ENERGY':
        #     LOG.info("Clustering files using ENERGY.")
        #     LOG.info("ENERGY threshold: %s kJ/mol", args.ewin)
        #     cluster_using_rmsd(args.energies, args.energy_window, args.max_rmsd, args.processors)
        #     LOG.info("Clustering complete!")
        else:
            LOG.error("Did not recognise clustering method. Options: RMSD, TORSIONS")

if __name__ == "__main__":
    main()
    
