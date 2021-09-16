#!/bin/sh

CURRENTPATH=$(pwd)
CFOLDPATH="${CURRENTPATH}/cfold"

rm -rf "${CFOLDPATH}"

echo "downloading the alphafold as ${CFOLDPATH}..."
git clone "https://github.com/deepmind/alphafold" "${CFOLDPATH}"
(
  cd "${CFOLDPATH}" || exit
  git checkout 1e216f93f06aa04aa699562f504db1d02c3b704c --quiet
)

echo "downloading ColabFold.py"
cd "${CFOLDPATH}" || exit
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py

echo "downloading ClusterFold scripts"
cd "${CFOLDPATH}" || exit
wget -qnc https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/fold_mmseqs2.py
wget -qnc https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/cfold.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/cfold_jackhmmer.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/cfold_mmseqs2.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/cfold_jackhmmer_slurm.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/cfold_mmseqs2_slurm.sh

echo "making outputs file"
cd "${CFOLDPATH}" || exit
if [ ! -d "$CFOLDPATH/outputs" ]; then
  mkdir "$CFOLDPATH/outputs"
fi

if [ ! -d "$CFOLDPATH/slurm_logs" ]; then
  mkdir "$CFOLDPATH/slurm_logs"
fi

echo "downloading AF2 parameters"
params_filename="alphafold_params_2021-07-14.tar"
params="${CFOLDPATH}/params"
if [ ! -d "$params" ]; then
  mkdir "$params"
fi
wget -P "$params" "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"
tar --extract --verbose --file="$params/$params_filename" --directory="$params" --preserve-permissions
rm "$params/$params_filename"

# colabfold patches
echo "Downloading several patches..."
mkdir "${CFOLDPATH}/patches"
cd "${CFOLDPATH}/patches" || exit
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/protein.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/config.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/model.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/modules.patch
wget -qnc https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/gpurelaxation.patch -O gpurelaxation.patch

# donwload reformat.pl from hh-suite
wget -qnc https://raw.githubusercontent.com/soedinglab/hh-suite/master/scripts/reformat.pl
# Apply multi-chain patch from Lim Heo @huhlim
cd "${CFOLDPATH}" || exit
patch -u alphafold/common/protein.py -i ./patches/protein.patch
patch -u alphafold/model/model.py -i ./patches/model.patch
patch -u alphafold/model/modules.py -i ./patches/modules.patch
patch -u alphafold/model/config.py -i ./patches/config.patch

# Install Miniconda3 for Linux
echo "installing Miniconda3 for Linux..."
cd "${CFOLDPATH}" || exit
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p "${CFOLDPATH}"/conda
rm Miniconda3-latest-Linux-x86_64.sh
cd ..

echo "creating conda environments with python3.7 as ${CFOLDPATH}/cfold-conda"
. "${CFOLDPATH}/conda/etc/profile.d/conda.sh"
export PATH="${CFOLDPATH}/conda/condabin:${. PATH}"
conda create -p "$CFOLDPATH"/cfold-conda python=3.7 -y
conda activate "$CFOLDPATH"/cfold-conda
conda update -y conda

echo "installing conda-forge packages"
conda install -c conda-forge python=3.7 cudnn==8.2.1.32 cudatoolkit==11.1.1 openmm==7.5.1 pdbfixer -y
echo "installing MSA binaries"
conda install -c bioconda hmmer hhsuite kalign2 -y
echo "installing alphafold and colabfold dependencies"
pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow-gpu==2.5.0
pip install jupyter matplotlib py3Dmol tqdm
pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

wget -q https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
mv stereo_chemical_props.txt ${CFOLDPATH}/alphafold/common

# Apply OpenMM patch.
echo "applying OpenMM patch..."
(cd ${CFOLDPATH}/cfold-conda/lib/python3.7/site-packages/ && patch -p0 <${CFOLDPATH}/docker/openmm.patch)

# Enable GPU-accelerated relaxation.
echo "enable GPU-accelerated relaxation..."
(cd ${CFOLDPATH} && patch -u alphafold/relax/amber_minimize.py -i patches/gpurelaxation.patch)

echo "installation of ClusterFold finished."
