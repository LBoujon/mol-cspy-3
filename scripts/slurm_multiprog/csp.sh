#!/bin/bash
echo "Starting CSP script on $(hostname) after 2s sleep in ${PWD}"
sleep 2
csp ${MOLS} -c ${CHARGES} -m ${MULTS} -a ${AXIS} \
    -g "${SPACEGROUPS}" -n ${NUMBER_STRUCTURES} --scheduler=file \
    --scheduler-file=${SCHEDULER_FILE}
