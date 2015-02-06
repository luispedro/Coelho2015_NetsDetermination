#!/bin/bash

# If set to 1 explicitly, then no warning will be printed

#NR_CPUS=1 

if ! which jug >/dev/null 2>/dev/null ; then
    echo "Jug (Python package) is not available."
    echo "Please check requirements in README file."
    exit 1
fi


if test -z "$NR_CPUS"; then
    echo "By default, this script uses a single processor. If you have more available, it is"
    echo "recommended that you edit this script and set the variable NR_CPUS to a larger number."
    echo "Otherwise, it may take a long time to get all results."
    echo
    NR_CPUS=1 
fi

function jug_execute {
    script=$1
    message=$2
    echo $message
    for i in $(seq $NR_CPUS); do
        jug execute $script --aggressive-unload &
    done
    wait
}

jug_execute jugfile.py "Running main analysis..."
jug_execute bernsen_thresholding.py "Testing simpler (threshold based) alternative..."
jug_execute compare.py "Computing human/human reliability..."
jug_execute rejected_region_counting.py.py "Measuring fraction of areas not accounted for..."
