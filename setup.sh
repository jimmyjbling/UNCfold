#!/bin/sh

CURRENTPATH=$(pwd)
UNCFOLDPATH="${CURRENTPATH}/uncfold"

rm -rf "${UNCFOLDPATH}"

echo "downloading the alphafold as ${UNCFOLDPATH}..."
git clone "https://github.com/deepmind/alphafold" "${UNCFOLDPATH}"
(cd "${UNCFOLDPATH}" || exit; git checkout 1e216f93f06aa04aa699562f504db1d02c3b704c --quiet)

echo "downloading ColabFold.py"
cd "${UNCFOLDPATH}" || exit
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/colabfold.py

echo "downloading UNCfold scripts"
cd "${UNCFOLDPATH}" || exit
wget -qnc https://raw.githubusercontent.com/jimmyjbling/UNCfold/main/fold_mmseqs2.py
wget -qnc https://raw.githubusercontent.com/jimmyjbling/UNCfold/main/uncfold.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/UNCfold/main/uncfold_jackhmmer.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/UNCfold/main/uncfold_mmseqs2.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/UNCfold/main/uncfold_jackhmmer_slurm.sh
wget -qnc https://raw.githubusercontent.com/jimmyjbling/UNCfold/main/uncfold_mmseqs2_slurm.sh

echo "making outputs file"
cd "${UNCFOLDPATH}" || exit
if [ ! -d "$UNCFOLDPATH/outputs" ]; then
  mkdir "$UNCFOLDPATH/outputs"
fi

if [ ! -d "$UNCFOLDPATH/slurm_logs" ]; then
  mkdir "$UNCFOLDPATH/slurm_logs"
fi

echo "downloading AF2 parameters"
params_filename="alphafold_params_2021-07-14.tar"
params="${UNCFOLDPATH}/params"
if [ ! -d "$params" ]; then
  mkdir "$params"
fi
wget -P "$params" "https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar"
tar --extract --verbose --file="$params/$params_filename" --directory="$params" --preserve-permissions
rm "$params/$params_filename"

# colabfold patches
echo "Downloading several patches..."
mkdir "${UNCFOLDPATH}/patches"
cd "${UNCFOLDPATH}/patches" || exit
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/protein.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/config.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/model.patch
wget -qnc https://raw.githubusercontent.com/sokrypton/ColabFold/main/beta/modules.patch
wget -qnc https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/gpurelaxation.patch -O gpurelaxation.patch

# donwload reformat.pl from hh-suite
wget -qnc https://raw.githubusercontent.com/soedinglab/hh-suite/master/scripts/reformat.pl
# Apply multi-chain patch from Lim Heo @huhlim
cd "${UNCFOLDPATH}" || exit
patch -u alphafold/common/protein.py -i ./patches/protein.patch
patch -u alphafold/model/model.py -i ./patches/model.patch
patch -u alphafold/model/modules.py -i ./patches/modules.patch
patch -u alphafold/model/config.py -i ./patches/config.patch

# Install Miniconda3 for Linux
echo "installing Miniconda3 for Linux..."
cd "${UNCFOLDPATH}" || exit
wget -q -P . https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p "${UNCFOLDPATH}"/conda
rm Miniconda3-latest-Linux-x86_64.sh
cd ..

echo "creating conda environments with python3.7 as ${UNCFOLDPATH}/uncfold-conda"
. "${UNCFOLDPATH}/conda/etc/profile.d/conda.sh"
export PATH="${UNCFOLDPATH}/conda/condabin:${PATH}"
conda create -p "$UNCFOLDPATH"/uncfold-conda python=3.7 -y
conda activate "$UNCFOLDPATH"/uncfold-conda
conda update -y conda

echo "installing conda-forge packages"
conda install -c conda-forge python=3.7 cudnn==8.2.1.32 cudatoolkit==11.1.1 openmm==7.5.1 pdbfixer -y
echo "installing MSA binaries"
conda install -c -y bioconda hmmer hhsuite kalign2
echo "installing alphafold dependencies"
pip install absl-py==0.13.0 biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0 numpy==1.19.5 scipy==1.7.0 tensorflow-gpu==2.5.0
pip install tqdm matplotlib py3dmol
pip install --upgrade jax jaxlib==0.1.69+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

wget -q https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
mv stereo_chemical_props.txt ${UNCFOLDPATH}/alphafold/common

# Apply OpenMM patch.
echo "applying OpenMM patch..."
(cd ${UNCFOLDPATH}/uncfold-conda/lib/python3.7/site-packages/ && patch -p0 < ${UNCFOLDPATH}/docker/openmm.patch)

# Enable GPU-accelerated relaxation.
echo "enable GPU-accelerated relaxation..."
(cd ${UNCFOLDPATH} && patch -u alphafold/relax/amber_minimize.py -i patches/gpurelaxation.patch)

echo "installation of UNCfold finished."