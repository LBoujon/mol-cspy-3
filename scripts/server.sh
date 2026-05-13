#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

export DIR=$PWD

function assert_set() {
    declare arg1="$1"
    if [ -z "${!arg1}" ]; then
        echo "ERROR: variable '\$$arg1' not set."
        exit 1
    fi
}
# DMACRYS timeout will need to be increased for:
# Spacegroups with more symm operations
# Larger molecules
# With large vdw_cutoffs
# More molecules (including cells which have been made P1)

#~~~~~~~~~~~~~~USER DEFINED PARAMETERS~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
export BACKUPDIR=""
export REMOTE_HOSTS="" # head node of the cluster to use
export SERVER_IP=""
export NUMBER_STRUCTURES=10000
export TIMEOUT="1200.0" # 600.0 1200.0
export VDW_CUTOFF="35.0" # 20.0 25.0
export REDISPORT=27272
export RABBITPORT=5672
export BROKERPORT="${RABBITPORT}"
export BACKENDPORT="${REDISPORT}"
export CSPY_LOC="$HOME/cspy"
export SG=(61 14 19 2 4 15 33 9 29 5)

for varname in BACKUPDIR REMOTE_HOSTS SERVER_IP NUMBER_STRUCTURES; do
assert_set "$varname"
done
for varname in SG BROKERPORT RABBITPORT CSPY_LOC; do
assert_set "$varname"
done

DMA="4.0" # for the old (quicker, less symmetric) DMA use 0, for the new version use 4.0
# Change the FF variable to the forcefield type desired, custom will require manual editing 
FF="F6311"
if [ ${FF} == "F631" ];then
### fit.pots ### 
export POT="${CSPY_LOC}/potentials/fit.pots"
export FORESHORTEN="n" # for the old (quicker, less symmetric) DMA use 0, for the new version use 4.0
export TYPE="F"
export FUNC="B3LYP"
export BASIS="6-31G**"
elif [ ${FF} == "F6311" ];then
### fit.pots ### 
export POT="${CSPY_LOC}/potentials/fit.pots"
export FORESHORTEN="n"
export TYPE="F"
export FUNC="B3LYP"
export BASIS="6-311G**"
elif [ ${FF} == "W631" ];then
### w99rev_631.pots ###
export POT="${CSPY_LOC}/potentials/w99rev_631_s.pots"
export FORESHORTEN="y"
export TYPE="W"
export FUNC="B3LYP"
export BASIS="6-31G**"
elif [ ${FF} == "W6311" ];then
### w99rev_6311.pots ###
export POT="${CSPY_LOC}/potentials/w99rev_6311_s.pots"
export FORESHORTEN="y"
export TYPE="W"
export FUNC="B3LYP"
export BASIS="6-311G**"
elif [ ${FF} == "C" ];then
### custom ###
export POT="`pwd`/dummyff.pots"
export FORESHORTEN="N"
export TYPE="C \ --custom_labels ../custom \ " # custom labels file must be given, this has not been tested so may need to be edited in manually
export FUNC=""
export BASIS=""
fi


for varname in FORESHORTEN TYPE FUNC BASIS FF; do
assert_set "$varname"
done

cd ${DIR}
for DIRS in $(find . -maxdepth 1 ! -path . -type d | sort -t _ -k 2 -g | xargs -r0 echo ); do
    cd ${DIR}/${DIRS}
    RUN=1
    echo "Working in $PWD"
    export RES=$(find . -maxdepth 1 -name '*.res' )
    export MULT=$(printf "../%s" *_mp.smult)
    echo "Starting server for $(basename ${RES}) (in ${PWD})"
    cspy-server -i ${RES} \
               -ip ${SERVER_IP} \
               --no-shutdown \
               --force_field_type ${TYPE} \
               --vdw_cutoff ${VDW_CUTOFF} \
               -sg ${SG[*]} \
               --max_iterations 2000 \
               -ns ${NUMBER_STRUCTURES} \
               -mf ${MULT} \
               --foreshorten_hydrogens ${FORESHORTEN} \
               --custom_atoms_anisotropy Br \
               -log "server-${RUN}.log" \
               -pf ${POT} \
               -cl 5 -zl 5 \
               --minimisations 3 \
               --wcal_ngcv "y" \
               --setup_off --spline_on \
               --dmacrys_timeout ${TIMEOUT} \
               --remote-hosts ${REMOTE_HOSTS} \
               --ignore-phonons --submit-workers \
               --number_workers 25
    # Ideally we either save all submitted pbs ids or we need the following command to work
    successful=$?
    cd ${DIR}
    if [ $successful -eq 0 ]; then
        scp -r ${DIR}/${DIRS} iridis4_a:${BACKUPDIR} && rm -rf ${DIR}/${DIRS}
    fi
done
