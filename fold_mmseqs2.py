import pickle
import re
import argparse
import time
import sys
import os
import warnings

import tensorflow as tf
import numpy as np

from absl import logging

from alphafold.common import protein
from alphafold.data import pipeline
from alphafold.data import templates
from alphafold.model import data
from alphafold.model import config
from alphafold.model import model
from alphafold.data.tools import hhsearch
from alphafold.relax import relax

import colabfold as cf

t0 = time.time()

device = "gpu"

########## PARSE ARGS ###############################
parser = argparse.ArgumentParser(description="No")

parser.add_argument('-f', '--fasta', type=str, metavar='', help='fasta file location')

parser.add_argument('-e', '--num_ensemble', default=1, type=int, metavar='', help='how many ensemble to use (1 or 8')
parser.add_argument('-a', '--amber', default=False, type=bool, metavar='', help='do you want to AMBER reduce')
parser.add_argument('-d', '--data_dir', default=None, type=str, metavar='', help='path to where data is stored')
parser.add_argument('-o', '--output_dir', default=None, type=str, metavar='', help='path for outputs')
parser.add_argument('-m', '--max_date', default="2100-01-01", type=str, metavar='', help='max date for templates')
parser.add_argument('-t', '--template', default=True, type=bool, metavar='', help='do you want to use templates')
parser.add_argument('-b', '--binaries', default=None, type=str, metavar='',
                    help='path to folder holding kalign and hmmrsearch binaries')

args = parser.parse_args()

######### SET PARAMS AND MAKE FILES #################

fasta_file = args.fasta

sequence = ""
with open(fasta_file, "r") as f:
    for line in f:
        if line.startswith(">"):
            continue
        sequence = sequence + line
sequence = re.sub("[^A-Z]", "", sequence.upper())

jobname = args.fasta.split("/")[-1][:-5]
jobname = re.sub(r'\W+', '', jobname)

print(jobname, "started")

print("using settings: ", args)

# prediction directory

if args.output_dir is None:
    output_dir = 'prediction_' + jobname + '_' + cf.get_hash(sequence)[:5]
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = args.output_dir

print("output dir located at: ", output_dir)

# delete existing files in working directory
for f in os.listdir(output_dir):
    os.remove(os.path.join(output_dir, f))

MIN_SEQUENCE_LENGTH = 16

a3m_file = os.path.join(output_dir, f"{jobname}.a3m")


def mk_mock_template(query_sequence):
    print("templates turned off... generating mock starting template")
    # since alphafold's model requires a template input
    # we create a blank example w/ zero input, confidence -1
    ln = len(query_sequence)
    output_templates_sequence = "-" * ln
    output_confidence_scores = np.full(ln, -1)
    templates_all_atom_positions = np.zeros((ln, templates.residue_constants.atom_type_num, 3))
    templates_all_atom_masks = np.zeros((ln, templates.residue_constants.atom_type_num))
    templates_aatype = templates.residue_constants.sequence_to_onehot(output_templates_sequence,
                                                                      templates.residue_constants.HHBLITS_AA_TO_ID)
    template_features = {'template_all_atom_positions': templates_all_atom_positions[None],
                         'template_all_atom_masks': templates_all_atom_masks[None],
                         'template_sequence': [f'none'.encode()],
                         'template_aatype': np.array(templates_aatype)[None],
                         'template_confidence_scores': output_confidence_scores[None],
                         'template_domain_names': [f'none'.encode()],
                         'template_release_date': [f'none'.encode()]}
    return template_features


def mk_template(a3m_lines, template_paths):
    print("generating template")
    template_featurizer = templates.TemplateHitFeaturizer(
        mmcif_dir=template_paths,
        max_template_date=args.max_date,
        max_hits=20,
        kalign_binary_path="/nas/longleaf/home/jwellni/miniconda3/envs/alphafold/bin/kalign",
        release_dates_path=None,
        obsolete_pdbs_path=None)

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path="/nas/longleaf/home/jwellni/miniconda3/envs/alphafold/bin/hhsearch",
        databases=[f"{template_paths}/pdb70"])

    hhsearch_result = hhsearch_pdb70_runner.query(a3m_lines)
    hhsearch_hits = pipeline.parsers.parse_hhr(hhsearch_result)
    templates_result = template_featurizer.get_templates(query_sequence=sequence,
                                                         query_pdb_code=None,
                                                         query_release_date=None,
                                                         hits=hhsearch_hits)
    return templates_result.features


def set_bfactor(pdb_filename, bfac, idx_res, chains):
    I = open(pdb_filename, "r").readlines()
    O = open(pdb_filename, "w")
    for line in I:
        if line[0:6] == "ATOM  ":
            seq_id = int(line[22:26].strip()) - 1
            seq_id = np.where(idx_res == seq_id)[0][0]
            O.write(f"{line[:21]}{chains[seq_id]}{line[22:60]}{bfac[seq_id]:6.2f}{line[66:]}")
    O.close()


def predict_structure(prefix, feature_dict, Ls, model_params, use_model, do_relax=False, random_seed=0):
    """Predicts structure using AlphaFold for the given sequence."""

    # Minkyung's code
    # add big enough number to residue index to indicate chain breaks
    idx_res = feature_dict['residue_index']
    L_prev = 0
    # Ls: number of residues in each chain
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i:] += 200
        L_prev += L_i
    chains = list("".join([ascii_uppercase[n] * L for n, L in enumerate(Ls)]))
    feature_dict['residue_index'] = idx_res

    # Run the models.
    plddts, paes = [], []
    unrelaxed_pdb_lines = []
    relaxed_pdb_lines = []

    print(do_relax)

    for model_name, params in model_params.items():
        t1 = time.time()
        if model_name in use_model:
            print(f"running {model_name}")
            # swap params to avoid recompiling
            # note: models 1,2 have diff number of params compared to models 3,4,5
            if any(str(m) in model_name for m in [1, 2]): model_runner = model_runner_1
            if any(str(m) in model_name for m in [3, 4, 5]): model_runner = model_runner_3
            model_runner.params = params

            processed_feature_dict = model_runner.process_features(feature_dict, random_seed=random_seed)
            prediction_result = model_runner.predict(processed_feature_dict)[0]
            unrelaxed_protein = protein.from_prediction(processed_feature_dict, prediction_result)
            unrelaxed_pdb_lines.append(protein.to_pdb(unrelaxed_protein))
            plddts.append(prediction_result['plddt'])
            paes.append(prediction_result['predicted_aligned_error'])

            if do_relax:
                # Relax the prediction
                print(f"relaxing {model_name}")
                amber_relaxer = relax.AmberRelaxation(max_iterations=0, tolerance=2.39,
                                                      stiffness=10.0, exclude_residues=[],
                                                      max_outer_iterations=20)
                relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)
                relaxed_pdb_lines.append(relaxed_pdb_str)
            print(f'{model_name} runtime: ', time.time() - t1)

    # rerank models based on predicted lddt
    lddt_rank = np.mean(plddts, -1).argsort()[::-1]
    out = {}
    print("reranking models based on avg. predicted lDDT")
    for n, r in enumerate(lddt_rank):
        print(f"model_{n + 1} {np.mean(plddts[r])}")

        unrelaxed_pdb_path = f'{prefix}_unrelaxed_model_{n + 1}.pdb'
        with open(unrelaxed_pdb_path, 'w') as f:
            f.write(unrelaxed_pdb_lines[r])
        set_bfactor(unrelaxed_pdb_path, plddts[r], idx_res, chains)

        if do_relax:
            relaxed_pdb_path = f'{prefix}_relaxed_model_{n + 1}.pdb'
            with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
            set_bfactor(relaxed_pdb_path, plddts[r], idx_res, chains)

        out[f"model_{n + 1}"] = {"plddt": plddts[r], "pae": paes[r]}
    return out


a3m_lines, template_paths = cf.run_mmseqs2(sequence, jobname, True, use_templates=args.template)
if template_paths is None:
    template_features = mk_mock_template(sequence)
else:
    template_features = mk_template(a3m_lines, template_paths)

with open(a3m_file, "w") as text_file:
    text_file.write(a3m_lines)

# parse MSA
msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)

from string import ascii_uppercase

# collect model weights
use_model = {}
if "model_params" not in dir(): model_params = {}
for model_name in ["model_1", "model_2", "model_3", "model_4", "model_5"][:5]:
    use_model[model_name] = True
    if model_name not in model_params:
        model_params[model_name] = data.get_model_haiku_params(model_name=model_name + "_ptm", data_dir=args.data_dir)
        if model_name == "model_1":
            model_config = config.model_config(model_name + "_ptm")
            model_config.data.eval.num_ensemble = 1
            model_runner_1 = model.RunModel(model_config, model_params[model_name])
        if model_name == "model_3":
            model_config = config.model_config(model_name + "_ptm")
            model_config.data.eval.num_ensemble = 1
            model_runner_3 = model.RunModel(model_config, model_params[model_name])

msas = [msa]
deletion_matrices = [deletion_matrix]

feature_dict = {
    **pipeline.make_sequence_features(sequence=sequence,
                                      description="none",
                                      num_res=len(sequence)),
    **pipeline.make_msa_features(msas=msas, deletion_matrices=deletion_matrices),
    **template_features
}
outs = predict_structure(os.path.join(output_dir, jobname), feature_dict,
                         Ls=[len(sequence)],
                         model_params=model_params, use_model=use_model,
                         do_relax=args.amber)

print("Total time", time.time() - t0)


