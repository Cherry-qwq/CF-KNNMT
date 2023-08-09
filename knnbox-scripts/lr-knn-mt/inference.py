from argparse import ArgumentParser
import os
import subprocess
import argparse
import sys
import re
from datetime import datetime
PROJECT_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
BASE_MODEL = f"{PROJECT_PATH}/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt"
ARCH_SUFFIX = "transformer_wmt19_de_en"

DEFAULT_DATASTORE_PATH = {
    ("vanilla", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("vanilla", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("vanilla", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("vanilla", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("adaptive", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("adaptive", "medical") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("adaptive", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("adaptive", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("pck", "it") : f"{PROJECT_PATH}/datastore/pck/it_dim64",
    ("pck", "koran") : f"{PROJECT_PATH}/datastore/pck/koran_dim64",
    ("pck", "law") : f"{PROJECT_PATH}/datastore/pck/law_dim64",
    ("pck", "medical") : f"{PROJECT_PATH}/datastore/pck/medical_dim64",
    ("lr", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("lr", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("lr", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("lr", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("lr_adaptive", "it") :  f"{PROJECT_PATH}/datastore/vanilla/it",
    ("lr_adaptive", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("lr_adaptive", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("lr_adaptive", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("lr_pck", "it") :  f"{PROJECT_PATH}/datastore/pck/it_dim64",
    ("lr_pck", "koran") : f"{PROJECT_PATH}/datastore/pck/koran_dim64",
    ("lr_pck", "law") : f"{PROJECT_PATH}/datastore/pck/law_dim64",
    ("lr_pck", "medical") : f"{PROJECT_PATH}/datastore/pck/medical_dim64"
}

DEFAULT_KNN_TEMPERATURE = {
    "it" : 10,
    "law" : 10,
    "koran" : 100,
    "medical" : 10   
}

DEFAULT_KNN_LAMBDA = {
    "it" : 0.7,
    "medical" : 0.8,
    "law" : 0.8,
    "koran" : 0.8
}

def get_dataset_path(ds):
    return f"{PROJECT_PATH}/data-bin/{ds}"

def get_dstore_path(method, ds):
    return f"{PROJECT_PATH}/datastore/{method}/ds"

def pjdir(x):
    return os.path.join(PROJECT_PATH, x)

def get_base_env(args):
    e = os.environ.copy()
    e["OMP_WAIT_POLICY"] = "PASSIVE"
    e["CUDA_VISIBLE_DEVICES"] = str(args.single_gpu_index)
    return e

def get_arch(m):
    if m == 'lr':
        return f"less_retrieve_knn_mt@{ARCH_SUFFIX}"
    else:
        return f"{m}_knn_mt@{ARCH_SUFFIX}"
    
def add_common_arguments(parser : argparse.ArgumentParser):
    parser.add_argument("--model", required=True, choices=['base', 'vanilla', 'adaptive', 'pck', 'lr', 'lr_adaptive', 'lr_pck'])
    parser.add_argument("--dataset", required=True, choices=['it', 'koran', 'law', 'medical'])
    parser.add_argument("--single-gpu-index", default=0)


if __name__ == "__main__":
    ps = ArgumentParser()
    add_common_arguments(ps)
    args = ps.parse_args()
    
    if args.model == 'base':
        cmd = [
            get_dataset_path(args.dataset),
            "--task", "translation",
            "--path", BASE_MODEL,
            "--dataset-impl", "mmap",
            "--beam", "4",
            "--lenpen", "0.6",
            "--max-len-a", "1.2",
            "--max-len-b", "10",
            "--source-lang", "de", 
            "--target-lang", "en", 
            "--gen-subset", "test",
            "--model-overrides", 
            "{'eval_bleu': False, 'required_seq_len_multiple':1, 'load_alignments': False}",
            "--max-tokens", "2048", 
            "--scoring", "sacrebleu", 
            "--tokenizer", "moses",
            "--remove-bpe"
        ]
        script = [sys.executable, pjdir("fairseq_cli/generate.py")]
        script.extend(cmd)
    else:
        cmd = [ 
            get_dataset_path(args.dataset),
            "--task", "translation",
            "--path", BASE_MODEL,
            "--dataset-impl", "mmap",
            "--beam", "4",
            "--lenpen", "0.6",
            "--max-len-a", "1.2",
            "--max-len-b", "10",
            "--source-lang", "de", 
            "--target-lang", "en", 
            "--gen-subset", "test",
            "--max-tokens", "2048", 
            "--scoring", "sacrebleu", 
            "--tokenizer", "moses",
            "--remove-bpe",
            "--user-dir", pjdir("knnbox/models"),
            "--arch", get_arch(args.model),
            "--knn-mode", "inference",
            "--knn-datastore-path", str(DEFAULT_DATASTORE_PATH[(args.model, args.dataset)]),
            "--knn-temperature", str(DEFAULT_KNN_TEMPERATURE[args.dataset])
        ]
        
        if args.model in ["lr", "vanilla"]:
            cmd.append("--knn-lambda")
            cmd.append(str(DEFAULT_KNN_LAMBDA[args.dataset]))
        
        if "lr" in args.model:
            cmd.append("--whether_retrieve_selector_path")
            cmd.append(pjdir(f"save-models/LRKNNMT/{args.dataset}/selector.pt"))
            
        if "adaptive" in args.model:
            cmd.append("--knn-combiner-path")
            cmd.append(pjdir(f"save-models/combiner/adaptive/{args.dataset}"))
            
        # about knn k
        if not "adaptive" in args.model:
            if not 'pck' in args.model:
                cmd.append("--knn-k")
                cmd.append("8")
            else:
                cmd.append("--knn-max-k")
                cmd.append("4")
                cmd.append("--knn-temperature-type")
                cmd.append("fixed")
        else:
            cmd.append("--knn-max-k")
            cmd.append("--knn-temperature-type")
            cmd.append("fixed")
            cmd.append("8")
        
        script = [sys.executable, pjdir("knnbox-scripts/common/generate.py")]
        script.extend(cmd)
    
    print(' '.join(script))    
    
    p = subprocess.Popen(script, env=get_base_env(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    p.wait()
    if p.returncode != 0:
        print(f"Error:\n {err.decode()}")
        exit(p.returncode)
    else:
        speed_info = err.decode().split("|")[-1].strip()
        print(speed_info)
        bleu_info = out.decode().split("Generate test with")[-1].strip()
        print(bleu_info)

        inference_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_results")
        os.makedirs(inference_results_path, mode=0o755, exist_ok=True)
        with open(f"{inference_results_path}/{args.model}-{args.dataset}-{datetime.now()}.txt", "w") as f:
            f.write(f"{speed_info}\n{bleu_info}")
        
        exit(0)