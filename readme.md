# ClusterFold
-----------------

ClusterFold (cfold) is a collection of shell/python scripts that allow you to quickly install and run Deepmind's [Alphafold](https://github.com/deepmind/alphafold) model on the UNC longleaf cluster (although any slurm managed cluster should also be able to work with this script).

Normally, the orginal alphafold setup can take a long time to run for medium sized protiens (500-700 residues). Thanks to the work done by the team behind [ColabFold](https://github.com/sokrypton/ColabFold), some ways to speed up alphafold can be implemnted to make runs faster. This collection implements both versions, the full orginal alphafold and alphafold with some of the tricks from colabfold. The difference between these two versions can be seen below:

| Version | data required | msa method | number of models | number of ensabmles | ABMER relaxation | run speed
| :-------- | -------  | --------- | ------- | --------- | ----------- | ----------|
| Alphafold | 2.2TB full OR 500GB reduced dataset | jackhmmer | 5 | 1 or 8 | required | slow
| Colabfold | 4 GB model params | [mmseqs2](https://github.com/soedinglab/MMseqs2) | 5 | 1 | optional | fast
-----------------

To install cfold, go into your home directory and collected the setup.sh file using:
<pre>
wget https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/setup.sh
</pre>

This will install alphafold, set up a miniconda python distribution with proper dempedencies installed, the model parameters and the scripts needed to run both versions described above in a new directory called cfold in the directory that setup.sh was run from. NOTE: Current version require 14GB of memory for install: 7GB for miniconda and 4GB for model parameters. If you lack the storage space, you can install paramters onto the scrach directory or tweak the miniconda install. Email me: jwellnitz@unc.edu with questions.

-----------------
To use cfold, go to the cfold directory and run cfold.sh:

<pre>
bash cfold.sh [msa_mode] [fasta_path] -a -t -d=<date>
</pre>

Usage: [msa_mode] [fasta_path] -a -t -d=<date>  
require arguments  
msa_mode:                          mmseqs2 (fast) or jackhmmer (slow). mmseqs2 will use colabfold approach, jackhmmer uses alphafold  
fasta_path:                        path to input sequence file in fasta format  
optional arguments  
-a | --amber_relax                 use amber relaxation (slow)  
-t | --template                    use templates  
-d | --max_template_date: <date>   date for template use cutoff in format YYYY-MM-DD (used only if the structure you are trying to predict exisits already)  

For example, I want to predict unknown_protien.fasta and I want to do it fast (so using the colabfold approach with mmseqs2). Since it hasn;t been sovled before I also was to use templates. I would run
  
<pre>
bash cfold.sh mmseqs2 path/to/unknown_protien.fasta -t
</pre>

Please cite

- Mirdita M, Ovchinnikov S and Steinegger M. ColabFold - Making protein folding accessible to all. 
bioRxiv (2021) doi: 10.1101/2021.08.15.456425

- Jumper et al. "Highly accurate protein structure prediction with AlphaFold."
Nature (2021) doi: 10.1038/s41586-021-03819-2
