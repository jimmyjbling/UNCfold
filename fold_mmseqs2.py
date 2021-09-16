import pickle
import re
import argparse
import time
import sys
import os
import warnings
import requests
import tarfile
import random
import jax
import hashlib

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

from string import ascii_uppercase, ascii_lowercase

t0 = time.time()

device = "gpu"

alphabet_list = list(ascii_uppercase + ascii_lowercase)

aatypes = set('ACDEFGHIKLMNPQRSTVWY')


def rm(x):
    '''remove data from device'''
    jax.tree_util.tree_map(lambda y: y.device_buffer.delete(), x)


def to(x, device="cpu"):
    '''move data to device'''
    d = jax.devices(device)[0]
    return jax.tree_util.tree_map(lambda y: jax.device_put(y, d), x)


def clear_mem(device="gpu"):
    '''remove all data from device'''
    backend = jax.lib.xla_bridge.get_backend(device)
    for buf in backend.live_buffers(): buf.delete()

def get_hash(x):
  return hashlib.sha1(x.encode()).hexdigest()

########## PARSE ARGS ###############################


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="No")

parser.add_argument('-f', '--fasta', type=str, metavar='', help='fasta file location')

parser.add_argument('-e', '--num_ensemble', default=1, type=int, metavar='', help='how many ensemble to use (1 or 8')
parser.add_argument('-a', '--amber', default=False, type=str2bool, metavar='', help='do you want to AMBER reduce')
parser.add_argument('-d', '--data_dir', default=None, type=str, metavar='', help='path to where data is stored')
parser.add_argument('-o', '--output_dir', default=None, type=str, metavar='', help='path for outputs')
parser.add_argument('-m', '--max_date', default="2100-01-01", type=str, metavar='', help='max date for templates')
parser.add_argument('-t', '--template', default=True, type=str2bool, metavar='', help='do you want to use templates')
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
    output_dir = 'prediction_' + jobname + '_' + get_hash(sequence)[:5]
    os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

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
        kalign_binary_path=os.path.join(args.binaries, "kalign"),
        release_dates_path=None,
        obsolete_pdbs_path=None)

    hhsearch_pdb70_runner = hhsearch.HHSearch(
        binary_path=os.path.join(args.binaries, "hhsearch"),
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
    print("predicting structure models")
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
        with open(os.path.join(output_dir, "model_stats.txt"), 'a') as f:
            f.write(f"{prefix.split('/')[-1]}_unrelaxed_model_{n + 1} plddt: {np.mean(plddts[r])} paes: {np.mean(paes[r])}\n")
        if do_relax:
            relaxed_pdb_path = f'{prefix}_relaxed_model_{n + 1}.pdb'
            with open(relaxed_pdb_path, 'w') as f: f.write(relaxed_pdb_lines[r])
            set_bfactor(relaxed_pdb_path, plddts[r], idx_res, chains)
            with open(os.path.join(output_dir, "model_stats.txt"), "a") as f:
                f.write(f"{prefix.split('/')[-1]}_relaxed_model_{n + 1} plddt: {np.mean(plddts[r])} paes: {np.mean(paes[r])}\n")
        out[f"model_{n + 1}"] = {"plddt": plddts[r], "pae": paes[r]}
    return out


def run_mmseqs2(x, prefix, use_env=True, use_filter=True,
                use_templates=False, filter=None, host_url="https://a3m.mmseqs.com"):
    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        res = requests.post(f'{host_url}/ticket/msa', data={'q': query, 'mode': mode})
        try:
            out = res.json()
        except ValueError:
            out = {"status": "UNKNOWN"}
        return out

    def status(ID):
        res = requests.get(f'{host_url}/ticket/{ID}')
        try:
            out = res.json()
        except ValueError:
            out = {"status": "UNKNOWN"}
        return out

    def download(ID, path):
        res = requests.get(f'{host_url}/result/download/{ID}')
        with open(path, "wb") as out: out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # compatibility to old option
    if filter is not None:
        use_filter = filter

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path): os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f'{path}/out.tar.gz'
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = sorted(list(set(seqs)))
    Ms = [N + seqs_unique.index(seq) for seq in seqs]

    # lets do it!
    if not os.path.isfile(tar_gz_file):
        while REDO:
            # Resubmit job until it goes through
            out = submit(seqs_unique, mode, N)
            while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                # resubmit
                time.sleep(5 + random.randint(0, 5))
                out = submit(seqs_unique, mode, N)

            # wait for job to finish
            ID, TIME = out["id"], 0
            while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                t = 5 + random.randint(0, 5)
                time.sleep(t)
                out = status(ID)
                if out["status"] == "RUNNING":
                    TIME += t
                # if TIME > 900 and out["status"] != "COMPLETE":
                #  # something failed on the server side, need to resubmit
                #  N += 1
                #  break

            if out["status"] == "COMPLETE":
                REDO = False

            if out["status"] == "ERROR":
                REDO = False
                raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence. If error persists, please try again an hour later.')

        # Download results
        download(ID, tar_gz_file)

    # prep list of a3m files
    a3m_files = [f"{path}/uniref.a3m"]
    if use_env: a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if not os.path.isfile(a3m_files[0]):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

            # templates
    if use_templates:
        templates = {}
        print("seq\tpdb\tcid\tevalue")
        for line in open(f"{path}/pdb70.m8", "r"):
            p = line.rstrip().split()
            M, pdb, qid, e_value = p[0], p[1], p[2], p[10]
            M = int(M)
            if M not in templates: templates[M] = []
            templates[M].append(pdb)
            if len(templates[M]) <= 20:
                print(f"{int(M) - N}\t{pdb}\t{qid}\t{e_value}")

        template_paths = {}
        for k, TMPL in templates.items():
            TMPL_PATH = f"{prefix}_{mode}/templates_{k}"
            if not os.path.isdir(TMPL_PATH):
                os.mkdir(TMPL_PATH)
                TMPL_LINE = ",".join(TMPL[:20])
                os.system(f"curl -s https://a3m-templates.mmseqs.com/template/{TMPL_LINE} | tar xzf - -C {TMPL_PATH}/")
                os.system(f"cp {TMPL_PATH}/pdb70_a3m.ffindex {TMPL_PATH}/pdb70_cs219.ffindex")
                os.system(f"touch {TMPL_PATH}/pdb70_cs219.ffdata")
            template_paths[k] = TMPL_PATH

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines: a3m_lines[M] = []
                a3m_lines[M].append(line)

    # return results
    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]

    if use_templates:
        template_paths_ = []
        for n in Ms:
            if n not in template_paths:
                template_paths_.append(None)
                print(f"{n - N}\tno_templates_found")
            else:
                template_paths_.append(template_paths[n])
        template_paths = template_paths_

    if isinstance(x, str):
        return (a3m_lines[0], template_paths[0]) if use_templates else a3m_lines[0]
    else:
        return (a3m_lines, template_paths) if use_templates else a3m_lines



a3m_lines, template_paths = run_mmseqs2(sequence, jobname, True, use_templates=args.template)
if template_paths is None:
    template_features = mk_mock_template(sequence)
else:
    template_features = mk_template(a3m_lines, template_paths)

with open(a3m_file, "w") as text_file:
    text_file.write(a3m_lines)

# parse MSA
msa, deletion_matrix = pipeline.parsers.parse_a3m(a3m_lines)

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
outs = predict_structure(output_dir, feature_dict,
                         Ls=[len(sequence)],
                         model_params=model_params, use_model=use_model,
                         do_relax=args.amber)

print("Total time", time.time() - t0)