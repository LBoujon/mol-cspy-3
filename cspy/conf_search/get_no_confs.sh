#!/bin/bash

#extract crest_conformers and crest_rotamers, get energies for each,


list=$1

if [ -f "conformer_numbers.csv" ]; then
    echo "conformer_numbers.csv exists."
    rm conformer_numbers.csv
fi

dir=`pwd`

for f in $list; do
    cd ${f}
    if [ -d "conformers" ] ; then
        echo "Directory \"conformers\" exists, deleting directory."
        rm -r conformers
    fi
    mkdir conformers
    if [ -f "crest_combined.xyz" ]; then
        echo "crest_combined.xyz exists."
    else
        echo "crest_combined.xyz not found. Creating file.."
        cat crest_conformers.xyz crest_rotamers.xyz >> crest_combined.xyz
    fi
    obabel crest_combined.xyz -O conformers/${f::-1}_.mol2 -m
    # energy is second line of xyz
    # make a table with the energy of each
    cd conformers
    echo "Retrieving energies.."
    for conf in *.mol2; do
        energy=`sed -n '2p' < $conf | awk '{print $1}'`
        echo "$conf $energy" >> energies.txt
    done
    python ~/progs/sort2.py
    #get the torsions file for each
    cp ../${f::-1}_torsions .
    echo "Reordering conformers by energy."
    bash ~/progs/reorder_v2.sh ${f::-1}
    echo "Begining to cluster.."
    bash ~/progs/cluster_torsions.sh ${f::-1}_
    cd uniques
    conf_num=`ls -1 *.xyz | wc -l`
    cd $dir
    echo "${f::-1},${conf_num}" >> conformer_numbers.csv
    echo "${f::-1} completed."
done
    
