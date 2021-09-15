#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=uncfold
#SBATCH -o "slurm_logs/$6.out"
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=50g
#SBATCH -t 02-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

module load cuda/11.2

# change me to full_dbs for full database search or casp14 for full search with 8 ensembles
PRESET="reduced_dbs"

bash uncfold_jackhmmer.sh -d $5 -o $(pwd)/outputs -m model_1,model_2,model_3,model_4,model_5 -f $1 -t $4 -y $(pwd)/uncfold-conda/bin/python3.7 -p $PRESET
EOT