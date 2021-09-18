# ClusterFold
-----------------

ClusterFold (cfold) is a collection of shell/python scripts that allow you to quickly install and run Deepmind's [Alphafold](https://github.com/deepmind/alphafold) model on the UNC longleaf cluster (although any slurm managed cluster should also be able to work with this script).

Normally, the orginal alphafold setup can take a long time to run for medium sized protiens (500-700 residues). Thanks to the work done by the team behind [ColabFold](https://github.com/sokrypton/ColabFold), some ways to speed up alphafold can be implemnted to make runs faster. This collection implements both versions, the full orginal alphafold and alphafold with some of the tricks from colabfold. The difference between these two versions can be seen below:

| Version | data required | msa method | number of models | number of ensabmles | ABMER relaxation | run speed
| :-------- | -------  | --------- | ------- | --------- | ----------- | ----------|
| Alphafold | 2.2TB full OR 500GB reduced dataset | jackhmmer | 5 | 1 or 8 | required | slow
| Colabfold | 4 GB model params | [mmseqs2](https://github.com/soedinglab/MMseqs2) | 5 | 1 | optional | fast
-----------------

One might be curious to see how the preformace between these to differ, so I rand some benchmarking with 9 random pdb file of residue lentgh between 200-300.
  
First, in runtime, the results showed there was a signficant difference between the two average runtime (p value of 9.512e-06) with the average for the colabfold appraoch taking 10 minutes and 17 seconds while the alphafold approach averaged 28 minutes and 11 seconds
  
In terms of prediction quaility, the average RMSD alignment between the prediction for each method was 0.5164 A. However one of the structure was a major oulier (5x0f at 2.139 A). Without that outlier the average RMSD was 0.3137 A
  
Lastly I compared each prediction to the real structure. The p value for the difference between the two approached not being zero was 0.3476 and for the colabfold method having better alignment scores better it was 0.1738. Overall there is no evidence to sugest one is better than the other in prediction quality

| PDB | Alphafold | Colabfold 
| :-------- | -------  | --------- |
| 3yn4 | 0.304867804050446 | 0.327671110630035
| 3nzc | 0.420691251754761 | 0.444653451442719
| 4ttq | 1.08367002010345 | 0.894327521324158
| 4u4h | 0.716829061508179 | 0.690542876720428
| 4u6a | 10.7000427246094 | 10.7304716110229
| 4w6e | 0.481924712657928 | 0.566877663135529
| 4wca | 0.383367419242859 | 0.393314033746719
| 4wda | 0.455938071012497 | 0.48809888958931
| 5x0f | 2.15102958679199 | 0.548271477222443
  
Take note that alphafold failed to fold 4u6a correct to any level. It is a good reminder to take caution and that alphafold is still just making predtions, and that those predictions can be very wrong.

-----------------

To install cfold, go into your home directory and collected the setup.sh file using:
<pre>
wget https://raw.githubusercontent.com/jimmyjbling/ClusterFold/main/setup.sh
</pre>

This will install alphafold, set up a miniconda python distribution with proper dempedencies installed, the model parameters and the scripts needed to run both versions described above in a new directory called cfold in the directory that setup.sh was run from. NOTE: Current version require 14GB of memory for install: 7GB for miniconda and 4GB for model parameters. If you lack the storage space, you can install paramters onto the scrach directory or tweak the miniconda install. Email me: jwellnitz@unc.edu with questions.

-----------------
To use cfold, go to the cfold directory and run cfold.sh:

<pre>
bash cfold.sh [msa_mode] [fasta_path] -a -t -d=(date)
</pre>

Usage: [msa_mode] [fasta_path] -a -t -d=<date>  
require arguments  
msa_mode:                          mmseqs2 (fast) or jackhmmer (slow). mmseqs2 will use colabfold approach, jackhmmer uses alphafold  
fasta_path:                        path to input sequence file in fasta format  
optional arguments  
-a | --amber_relax                 use amber relaxation (slow)  
-t | --template                    use templates  
-d | --max_template_date: (date)   date for template use cutoff in format YYYY-MM-DD (used only if the structure you are trying to predict exisits already)  

For example, I want to predict unknown_protien.fasta and I want to do it fast (so using the colabfold approach with mmseqs2). Since it hasn;t been sovled before I also was to use templates. I would run
  
<pre>
bash cfold.sh mmseqs2 path/to/unknown_protien.fasta -t
</pre>  
-----------------
Now I might want to run the protien unknown protien with the full version of alphafold, as it might be slightly better. I would run the following.
Note in this case the -a or -t isn't required, as the jackhmmer mode will always amber relax and use templates, but you can include them it won't change anything (see table above)
<pre>
bash cfold.sh jackhmmer path/to/unknown_protien.fasta -a -t
</pre>
-----------------
If I wanted to run a already known protien 6y4f (maybe to see for myself how good alphafold's predictions are) I would want to aviod using templates. But I want to amber relax so I run
  
<pre>
bash cfold.sh mmseqs2 path/to/6y4f.fasta -a
</pre>  
  
 (maybe to see for myself how good alphafold's predictions are) I would want to aviod using templates. But I want to amber relax so I run

Please cite

- Mirdita M, Ovchinnikov S and Steinegger M. ColabFold - Making protein folding accessible to all. 
bioRxiv (2021) doi: 10.1101/2021.08.15.456425

- Jumper et al. "Highly accurate protein structure prediction with AlphaFold."
Nature (2021) doi: 10.1038/s41586-021-03819-2
