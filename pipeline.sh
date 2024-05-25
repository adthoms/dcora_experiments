#!/usr/bin/env bash

set -e  # exit on error

if [[ "$#" -ne 3 ]]; then
    echo "Usage: [.pyfg file] [(CORA) .tum file] [(DCORA) .tum file]"
    exit 1
fi

# printf "\033c" resets the output
function log { printf "\033c"; echo -e "\033[32m[$BASH_SOURCE] $1\033[0m"; }
function echo_and_run { echo -e "\$ $@" ; read input; "$@" ; read input; }

# always run in script directory
# parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
# cd "$parent_path"

log "Generate ground truth from .pyfg file"
echo_and_run python3 pyfg_to_tum.py $1 temp_gt.tum

log "Compare TUM files"
echo_and_run evo_traj tum $2 --ref=temp_gt.tum -p --plot_mode.xz