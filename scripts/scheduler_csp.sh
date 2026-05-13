#!/bin/bash
${SCHEDULER_CMD} 2>&1 > logs/scheduler.log &
sleep 2
${CSP_CMD}
