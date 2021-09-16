#!/bin/bash

usage() {
        echo ""
        echo "Usage: $0 <OPTIONS>"
        echo "-p <python> path to python3 binary"
        echo "-d <param_dir>     Path to directory of supporting data"
        echo "-o <output_dir>   Path to a directory that will store the results."
        echo "-f <fasta_path>   Path to a FASTA file containing one sequence"
        echo "-t <max_template_date> Maximum template release date to consider (ISO-8601 format - i.e. YYYY-MM-DD). Important if folding historical test sets"
        echo "-m <mode>         1 for template, 0 for no template"
        echo "-e <ensemble>     Number of ensembles (1 or 8)"
        echo "-a <amber>        Relax with amber?"
        echo ""
        exit 1
}

while getopts ":d:o:m:f:t:e:a" i; do
        case "${i}" in
        d)
                param_dir=$OPTARG
        ;;
        o)
                output_dir=$OPTARG
        ;;
        m)
                mode=$OPTARG
        ;;
        f)
                fasta_path=$OPTARG
        ;;
        t)
                max_template_date=$OPTARG
        ;;
        a)
                amber=$OPTARG
        ;;
        e)
                ensemble=$OPTARG
        ;;
        p)
                python_bin=$OPTARG
        ;;
        esac
done

use_gpu=true
gpu_devices=0

# Parse input and set defaults
if [[ "$param_dir" == "" || "$fasta_path" == "" ]] ; then
    usage
fi

if [[ "$mode" == "" ]] ; then
    mode=1
fi

if [[ "$ensemble" == "" ]] ; then
    ensemble=1
fi

if [[ "$amber" == "" ]] ; then
    amber=0
fi

if [[ "$max_template_date" == "" ]] ; then
    max_template_date="2100-01-01"
fi

jobname=$(basename $fasta_path)

if [ ! -d "$output_dir/$jobname" ]; then
  mkdir "$output_dir/$jobname"
fi

result_dir="$output_dir/$jobname"

# Export ENVIRONMENT variables and set CUDA devices for use
# CUDA GPU control
export CUDA_VISIBLE_DEVICES=-1
if [[ "$use_gpu" == true ]] ; then
    export CUDA_VISIBLE_DEVICES=0

    if [[ "$gpu_devices" ]] ; then
        export CUDA_VISIBLE_DEVICES=$gpu_devices
    fi
fi

# TensorFlow control
export TF_FORCE_UNIFIED_MEMORY='1'

# JAX control
export XLA_PYTHON_CLIENT_MEM_FRACTION='4.0'

# Run AlphaFold with required parameters
./cfold-conda/bin/python3.7 ./fold_mmseqs2.py -d $param_dir -f $fasta_path -o $result_dir -m $max_template_date -a $amber -e $ensemble -t $mode -b ./cfold-conda/bin

