#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=uncfold
#SBATCH -o "./slurm_logs/$5.out"
#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=50g
#SBATCH -t 02-00:00:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1

module load cuda/11.2

bash uncfold_mmseqs2.sh "-d $(pwd)/params -o $(pwd)/outputs -f $1 -t $4 -m $3 -e 1 -a $2 -p $(pwd)/uncfold-conda/bin/python3.7"
EOT