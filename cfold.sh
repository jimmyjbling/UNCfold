#!/bin/bash

CURRENTPATH=$(pwd)

usage() {
        echo ""
        echo "Please make sure msa_mode and fasta_path are passed (in that order)"
        echo "Usage: [msa_mode] [fasta_path] -a -t -m"
        echo "---------------------------------------------------------------------------"
        echo "require arguments"
        echo "msa_mode:                          mmseqs2 (fast) or jackhmmer (slow)"
        echo "fasta_path:                        path to input sequence file in fasta format"
        echo "optional arguments"
        echo "-a | --amber_relax                 use amber relaxation (slow)"
        echo "-t | --template                    use templates"
        echo "-d | --max_template_date: <date>   date for template use cutoff in format YYYY-MM-DD"
        exit 1
}

DATADIR=""

AMBER=0
TEMPLATE=0
MAX_TEMPLATE_DATE="2100-01-01"

PARAMS=""
while (( "$#" )); do
  case "$1" in
    -a|--amber_relax)
      AMBER=1
      shift
      ;;
    -t|--template)
      TEMPLATE=1
      shift
      ;;
    -d|--max_template_date)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        MAX_TEMPLATE_DATE=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -*|--*=)
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *)
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

eval set -- "$PARAMS"

if [[ "$1" == "" || "$2" == "" ]] ; then
    usage
fi

if [[ "$2" == "" ]] ; then
  echo "Error: pass a path to a input fasta file"
  exit 1
else
  JOBNAME=$(basename "$2")
fi

if [[ "$DATADIR" == "" && "$1" == "jackhmmer" ]] ; then
    echo "Error: jackhmmer alphafold dataset directory not set. Edit line 20 of cfold.sh to path to data directory"
    exit 1
fi

if  [[ "$1" == "mmseqs2" ]] ; then
    sbatch $CURRENTPATH/cfold_mmseqs2_slurm.sh -f $2 $AMBER $TEMPLATE $MAX_TEMPLATE_DATE $JOBNAME
    exit 0
fi

if [[ "$1" == "jackhmmer" ]] ; then
    sbatch $CURRENTPATH/cfold_jackhmmer_slurm.sh -f $2 $AMBER $TEMPLATE $MAX_TEMPLATE_DATE $DATADIR $JOBNAME
    exit 0
fi

echo "Error: msa mode not regonized: $1. accepted values are mmseqs2 or jackhmmer"
exit 1

