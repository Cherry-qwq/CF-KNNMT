from argparse import ArgumentParser
import os
import subprocess
import argparse
import sys
import re
import numpy as np
from datetime import datetime
from comet import download_model, load_from_checkpoint
PROJECT_PATH = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-3])
BASE_MODEL = f"{PROJECT_PATH}/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt"
ARCH_SUFFIX = "transformer_wmt19_de_en"

DEFAULT_DATASTORE_PATH = {
    ("vanilla", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("vanilla", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
    ("vanilla", "law") : f"{PROJECT_PATH}/datastore/vanilla/law",
    ("vanilla", "medical") : f"{PROJECT_PATH}/datastore/vanilla/medical",
    ("adaptive", "it") : f"{PROJECT_PATH}/datastore/vanilla/it",
    ("adaptive", "koran") : f"{PROJECT_PATH}/datastore/vanilla/koran",
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
    "it" : 0.7,#0.7
    "medical" : 0.8,#0.8
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
    #e["CUDA_VISIBLE_DEVICES"] = "6"#str(args.single_gpu_index)
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
    parser.add_argument("--run-3-time", default=False, action='store_true')
    parser.add_argument("--no-translation-loss", default=False, action='store_true')
    parser.add_argument("--test-knn-overhead", default=False, action='store_true')
    parser.add_argument("--knn-k", default=None)

def extract_translations(output):
    lines = output.strip().split('\n')
    hypothesis_translations = {}
    reference_translations = {}

    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 3:
            idx, content = parts[0], parts[-1]
            if idx.startswith('H-') or idx.startswith('D-'):
                hypothesis_translations[idx[2:]] = content
            elif idx.startswith('T-'):
                reference_translations[idx[2:]] = content
    
    # 确保每个假设翻译都有一个对应的参考翻译
    translations = [
        (hypothesis_translations[idx], reference_translations[idx])
        for idx in hypothesis_translations if idx in reference_translations
    ]
    
    return translations
import ast
def calculate_comet_score(file_path):

    with open(file_path, 'r') as file:
        content = file.read()
    lines = ast.literal_eval(content)
# Initialize the lists
    SList = []
    DList = []
    TList = []

# Filter the lines
    for line in lines:

        if line.startswith('D-'):
            DList.append(line.strip())
        elif line.startswith('T-'):
            TList.append(line.strip())
        elif line.startswith('S-'):
            SList.append(line.strip())

    # with open(os.path.join(save_path,'D.txt'),'w') as fd:
    #     fd.write(str(DList))
    # with open(os.path.join(save_path,'T.txt'),'w') as fd:
    #     fd.write(str(TList))    
    # with open(os.path.join(save_path,'S.txt'),'w') as fd:
    #     fd.write(str(SList))

    # 打印或处理评估分数
    # for score in scores:
    #     print(score)
    data = []

# 遍历列表元素
    for s, d, t in zip(SList, DList, TList):
        # 从每个元素中提取所需信息
        src = s.split('\t')[-1]  # 源句
        mt = d.split('\t')[-1]  # 机器翻译输出
        ref = t.split('\t')[-1]  # 参考翻译
        with open(os.path.join("/data/qirui/z-testdata",'vanillamodel_koran.txt'),'a') as fd:
            fd.write(str(mt)+"\n")
        # 创建字典并添加到结果列表
        entry = {
            "src": src,
            "mt": mt,
            "ref": ref
        }
        data.append(entry)



    return data

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
        if args.knn_k:
            cmd.append("--knn-k")
            cmd.append(str(args.knn_k))
        
        if args.model in ["lr", "vanilla"]:
            cmd.append("--knn-lambda")
            cmd.append(str(DEFAULT_KNN_LAMBDA[args.dataset]))
        
        if "lr" in args.model:
            cmd.append("--whether_retrieve_selector_path")
            #cmd.append(pjdir(f"save-models/LRKNNMT/{args.dataset}/selector.pt"))
            cmd.append(pjdir(f"save-models/LRKNNMT/{args.dataset}/selector.pt") if not args.no_translation_loss else pjdir(f"save-models/LRKNNMT/{args.dataset}/selector_no_translation_loss.pt"))         
            
        if "adaptive" in args.model:
            cmd.append("--knn-combiner-path")
            cmd.append(pjdir(f"save-models/combiner/adaptive/{args.dataset}"))
        
        if 'pck' in args.model:
            cmd.append("--knn-combiner-path")
            cmd.append(pjdir(f"save-models/combiner/pck/{args.dataset}_dim64"))
            
        # about knn k
        if args.knn_k is None:
            if not "adaptive" in args.model:
                if not 'pck' in args.model:
                    cmd.append("--knn-k")
                    cmd.append("8")#8
                else:
                    cmd.append("--knn-max-k")
                    cmd.append("4")
                    cmd.append("--knn-temperature-type")
                    cmd.append("fixed")
            else:
                cmd.append("--knn-max-k")
                cmd.append("8")
                cmd.append("--knn-temperature-type")
                cmd.append("fixed")
        #创建一个完整的命令行指令，用于启动并运行 knnbox-scripts/common/generate.py 脚本，并向其传递所有必要的参数
        script = [sys.executable, pjdir("knnbox-scripts/common/generate.py")]
        script.extend(cmd)
    
    print(' '.join(script))    
    
    
    if not args.run_3_time:
        #执行之前构建的 script 命令行指令。
        p = subprocess.Popen(script, env=get_base_env(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()

        output_str = out.decode()
        lines = output_str.split('\n')
# #####

        # local_model_path = "/data/qirui/xlm-roberta-large"
        # comet_model = load_from_checkpoint(local_model_path)
        out_path = f"test_result/{args.dataset}.txt"
        with open(out_path,"w" ) as file:
            file.write(str(lines))
        model_path = download_model("wmt20-comet-da")
        comet_model = load_from_checkpoint(model_path)
        # translations = extract_translations(output_str)
        data = calculate_comet_score(out_path)
        #scores = comet_model.predict(S,D,T)
    #     data = [{
    #     "src": "This is a test sentence.",
    #     "mt": "这是一个测试句子。",
    #     "ref": "这是一句测试句子。"
    # }]
        scores = comet_model.predict(samples = data)
        comet_score = scores[1]
        #print("Average COMET score:", comet_score)  # 输出每个翻译句子的质量分数
        print("Average COMET score:", round(comet_score * 100, 2))
        # # 打印评分结果
        # for (hyp, ref), score in zip(translations, scores):
        #     print(f"Hypothesis: {hyp}\nReference: {ref}\nCOMET Score: {score}\n")



# ######
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
                
                if args.no_translation_loss:
                    f.write("\nNote: The selector was trained without translation loss.")
            
            exit(0)
    else:
        bleu = None
        times = []
        tps = []
        knn_time = []
        for li in range(3):
            p = subprocess.Popen(script, env=get_base_env(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            p.wait()
            
            if p.returncode != 0:
                print(f"Error:\n {err.decode()}")
                exit(p.returncode)
            else:
                out_info = out.decode()
                
                if bleu is None:
                    #import pdb
                    #pdb.set_trace()
                    bleu_info = out_info.split("\n")[-3].split("Generate test with")[-1].strip()
                   # bleu = float(re.match(r"BLEU = ([\d\.]+)", bleu_info)[1])
                    bleu = float(bleu_info.split(":")[1].split(' ')[3])
                    print(f"BLEU = {bleu}")
                    
                if args.test_knn_overhead:
                    knn_retrieve_info = out_info.split("\n")[-2]
                    knn_time.append( float(knn_retrieve_info.split('=')[-1].strip().split('s')[0]) )                
            
                speed_info = err.decode().split("|")[-1].strip()
                #print(speed_info)
                times.append(float(speed_info.split('in')[1].split(' ')[1].split('s')[0]))
                tps.append(float(speed_info.split("/s")[-2].split(' ')[-2]))
                if args.test_knn_overhead:
                    print(f"Run #{li+1}, inference time = {times[-1]}, {tps[-1]} tokens/s, KNN overhead = {knn_time[-1]}s")
                else:
                    print(f"Run #{li+1}, inference time = {times[-1]}, {tps[-1]} tokens/s")
                
                

        inference_results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference_results")
        os.makedirs(inference_results_path, mode=0o755, exist_ok=True)
        with open(f"{inference_results_path}/{args.model}-{args.dataset}-{datetime.now()}.txt", "w") as f:
            f.write(f"BLEU = {bleu}\nEach inference time in seconds = {' '.join(list(map(str,times)))}\nAverage inference time = {sum(times)/len(times)}s")
            f.write(f"\nEach inference speed(tokens/s) = {' '.join(list(map(str, tps)))}\nAverage inference speed = {sum(tps)/len(tps)} tokens/s")
            
            if args.test_knn_overhead:
                f.write(f"\nEach KNN retrieving time = {' '.join(list(map(str, knn_time)))}\nAverage KNN retrieving time = {sum(knn_time)/len(knn_time)}s")
            
        
        
        exit(0)

            