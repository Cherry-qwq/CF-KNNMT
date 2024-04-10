r""" 
This file is copied from fairseq_cli/validate.py.
knnbox made 2 major changes:

change 1. We modified the part of parsing args so that is
can parse the arch specified on the cli instead of directly
using the arch inside checkpoint.

change 2. we add codes about `saving datastore vals`, `dump datastore`, etc. 
"""
# from fairseq.models import (
#     register_model,
#     register_model_architecture,
# )

# from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner
import logging
from collections import Counter
import os
import sys
from itertools import chain
from itertools import islice
import torch
import json
import math
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar

## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
from knnbox.datastore import Datastore, GreedyMergeDatastore, PckDatastore
from knnbox.common_utils import filter_pad_tokens, global_vars
import numpy as np
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.validate")


def main(args, override_args=None):
    utils.import_user_module(args)

    assert (
        args.max_tokens is not None or args.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)

    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]
    
    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    # Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    knn_type = args.arch.split("@")[0]
    dictionary_len=len(task.tgt_dict)
    if "datastore" not in global_vars():
        # create suitable datastore class if not exists
        if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt", "kernel_smoothed_knn_mt", "vanilla_knn_mt_visual", "robust_knn_mt"]:
            global_vars()["datastore"] = Datastore(path=args.knn_datastore_path)
            
        if knn_type == "greedy_merge_knn_mt":
            global_vars()["datastore"] = GreedyMergeDatastore(path=args.knn_datastore_path)
        if knn_type == "pck_knn_mt":
            global_vars()["datastore"] = PckDatastore(
                path=args.knn_datastore_path,
                reduction_network_input_dim=args.decoder_embed_dim,
                reduction_network_output_dim=args.knn_reduct_dim,
                dictionary_len=len(task.tgt_dict),
                )
    datastore = global_vars()["datastore"]
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    #对于每一个类型的数据集
    for subset in args.valid_subset.split(","):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception("Cannot find dataset: " + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.batch_size,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
            data_buffer_size=args.data_buffer_size,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=("tqdm" if not args.no_progress_bar else "simple"),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt", "greedy_merge_knn_mt", "kernel_smoothed_knn_mt", "plac_knn_mt", "robust_knn_mt"]:
                ###向datastore添加数据：
                #过滤出非填充（non-pad）标记，并生成相应的掩码（mask）。
                non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                datastore["vals"].add(non_pad_tokens)
                datastore.set_pad_mask(mask)
            
            elif knn_type == "pck_knn_mt":
                non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                datastore["vals"].add(non_pad_tokens)
                datastore.set_pad_mask(mask)
                datastore.set_target(sample["target"])

            elif knn_type == "vanilla_knn_mt_visual":
                non_pad_tokens, mask = filter_pad_tokens(sample["target"])
                datastore["vals"].add(non_pad_tokens)
                datastore.set_pad_mask(mask)
                # get the key-value pair related sentence_ids 
                target_len = mask.sum(dim=-1)
                sentence_ids = []
                for idx, sentence_id in enumerate(sample["id"].cpu().numpy()):
                    sentence_ids += [sentence_id]*target_len[idx]
                sentence_ids = np.array(sentence_ids, dtype=int)
                # get the key-value pair related token_postions
                token_positions = []
                for len_ in target_len:
                    token_positions += [i for i in range(len_)]
                token_positions = np.array(token_positions, dtype=int)
                # add them to datastore
                datastore["sentence_ids"].add(sentence_ids)
                datastore["token_positions"].add(token_positions)
            ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end

            _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
            progress.log(log_output, step=i)
            log_outputs.append(log_output)
        #######在这里写！
        #print("!!!注意这里！",datastore["vals"][:10])
        #这段代码用于在分布式训练环境中收集并聚合每个 GPU 上的验证输出（log_outputs），计算性能指标，然后输出聚合后的指标到控制台。这有助于监控整体分布式训练的性能。
        if args.distributed_world_size > 1:
            log_outputs = distributed_utils.all_gather_list(
                log_outputs,
                max_size=getattr(args, "all_gather_list_size", 16384),
            )
            log_outputs = list(chain.from_iterable(log_outputs))

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()

        progress.print(log_output, tag=subset, step=i)
    #get_CF(args.knn_datastore_path)
    #FAISS与剪枝:
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # release memory to make sure we have enough gpu memory to build faiss index
    del model, task, progress, criterion, dataset
    if use_cuda:
        torch.cuda.empty_cache()    # release gpu memory

    if knn_type in ["vanilla_knn_mt", "adaptive_knn_mt", "kernel_smoothed_knn_mt", "vanilla_knn_mt_visual", "plac_knn_mt", "robust_knn_mt"]:
        datastore.dump()    # dump to disk
        datastore.build_faiss_index("keys", use_gpu=(not args.build_faiss_index_with_cpu))   # build faiss index
    elif knn_type == "greedy_merge_knn_mt":
        datastore.dump() # dump the un-pruned datastore to disk
        datastore.build_faiss_index("keys", do_pca=args.do_pca, pca_dim=args.pca_dim, use_gpu=(not args.build_faiss_index_with_cpu)) # build faiss index with pre-PCA operation for pruned datastore
        if args.do_merge:
            datastore.prune(merge_neighbors=args.merge_neighbors_n) # prune the datastore. search n neighbors when do greedy merge
            datastore.dump() # dump the pruned datastore to disk
            datastore.build_faiss_index("keys", do_pca=args.do_pca, pca_dim=args.pca_dim, use_gpu=(not args.build_faiss_index_with_cpu)) # build faiss index for un-pruned datastore

    elif knn_type == "pck_knn_mt":
        datastore.dump() # dump the un-pruned datastore to disk
    get_CF(args.knn_datastore_path)
    select_k(args.knn_datastore_path,dictionary_len)
    cal_dis(args.knn_datastore_path,dictionary_len)

    #在这个地方再遍历一次所有keys，确定每个keys的k的集合，加入进klist数组，然后选出最具代表性的k值保存进文件中
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
def cal_dis(ds_path,dic_len):
    datastore = Datastore.load(f"{ds_path}", load_list=["keys","vals"])
    vals_np= np.array(datastore['vals'].data, dtype=np.int64)
    retriever = Retriever(datastore=datastore, k=2)
    combiner = Combiner(lambda_=0.5,
                     temperature= 10 , probability_dim=dic_len, datastore_path= ds_path)
    dis_list = []
    result = {}
    keys_tensor = torch.tensor(datastore['keys'].data, dtype=torch.float32)

    tag = 0
    batch = 256000  # 选择合适的批大小(512000可以)
    dmax = 0
    dmin = 10000
    for it in range(0, len(keys_tensor), batch):
        print(f"正在处理第{it // batch +1}轮，共{math.ceil(len(keys_tensor) / batch)}轮：")
        batch_keys_tensor = keys_tensor[it:it+batch]
        if torch.cuda.is_available():
            batch_keys_tensor = batch_keys_tensor.cuda()
        
        query = batch_keys_tensor.detach()
        retriever.retrieve(query, return_list=["distances"])
        
        dis_m = retriever.results['distances']
        print(dis_m.size())#(batch_size,k)
        dis = dis_m[:, 1:]
        for item in dis:
            for i in item:
                dis_list.append(i)
                if i < dmin:
                    dmin = i
                if i > dmax:
                    dmax = i
        torch.cuda.empty_cache()
    print(len(dis_list))
    
    result["dmin"] = dmin.item() if isinstance(dmin, torch.Tensor) else dmin
    result["dmax"] = dmax.item() if isinstance(dmax, torch.Tensor) else dmax
    total = len(dis_list)
    result["vmax"] = total ** (1/3)
    tf = open(os.path.join(ds_path,"parameters.json"), "w")
    json.dump(result,tf)
    tf.close()
    print("已将收集数据保存至目录。///")



def select_k(ds_path,dic_len):
    datastore = Datastore.load(f"{ds_path}", load_list=["keys","vals"])
    vals_np= np.array(datastore['vals'].data, dtype=np.int64)
    shape = vals_np.shape[0]
    print("shape:",shape)
    klist = [3,5,6,7,8,9,10,11,13,15]
    ktag = []
    kfinal = []
    retriever = Retriever(datastore=datastore, k=8)
    combiner = Combiner(lambda_=0.5,
                     temperature= 10 , probability_dim=dic_len,datastore_path=ds_path)
    
    # # 获取数据的形状和数据类型
    # shape = datastore['keys'].shape
    # dtype = datastore['keys'].dtype
    # # 获取实际的 Numpy 数组
    # data_array = datastore['keys'].data
    # # 输出形状和数据类型
    # print(f"Shape: {shape}, Dtype: {dtype}")
    # # 输出前几个元素
    # print("Data (first 10 elements):", data_array[:10])
    k_dic = {}
    for i in range(shape):
        k_dic[i] = []
    keys_tensor = torch.tensor(datastore['keys'].data, dtype=torch.float32)

    tag = 0
    batch = 512000  # 选择合适的批大小(512000可以)
    for it in range(0, len(keys_tensor), batch):
        print(f"正在处理第{it // batch +1}轮，共{math.ceil(len(keys_tensor) / batch)}轮：")
        batch_keys_tensor = keys_tensor[it:it+batch]
        if torch.cuda.is_available():
            batch_keys_tensor = batch_keys_tensor.cuda()
        


   
    # if torch.cuda.is_available():
    #     keys_tensor = keys_tensor.cuda()

        for kl in klist:
            query = batch_keys_tensor.detach()
            retriever.retrieve(query, return_list=["vals", "distances"],k = kl + 1)
            vals_m = retriever.results['vals']
            dis_m = retriever.results['distances']

            vals = vals_m[:, 1:]
            dis = dis_m[:, 1:]

            if vals.dim() == 2:
        # 增加一个维度，假设序列长度（S）为 1
                #vals = vals.unsqueeze(1)
                S, K = vals.size()
                print(S,K)#S = 96
                print(vals.size())
                batch_size = 96
                new_vals = vals[:S - S % batch_size ].reshape(batch_size, -1, K)
                new_dis = dis[:S - S % batch_size ].reshape(batch_size, -1, K)
                new_vals2 = vals[S - S % batch_size : ].reshape(1, -1, K)
                new_dis2 = dis[S - S % batch_size : ].reshape(1, -1, K)
        # 现在 vals 的形状应该是 [B, S, K]
            tag = it
            for i,item in enumerate(new_vals):
                new_v = item.unsqueeze(0)
                itemd = new_dis[i].unsqueeze(0)
                retriever.results['vals'] = new_v
                retriever.results['distances'] = itemd
                knn_prob =combiner.get_knn_prob(**retriever.results, device="cuda:0")
                
                _, indices = torch.max(knn_prob, dim=2)
                indices = indices.squeeze()
                if indices.dim() == 0:
                    #continue
                    indices = indices.unsqueeze(0)
                
                for item in indices:
                    if item.item() == vals_np[tag]:
                        k_dic[tag].append(kl)
                    tag += 1
                    
                #print(indices,indices.size())
            retriever.results['vals'] = new_vals2
            retriever.results['distances'] = new_dis2
            knn_prob =combiner.get_knn_prob(**retriever.results, device="cuda:0")
            _, indices = torch.max(knn_prob, dim=2)
            print("results:",retriever.results['vals'])
            print("distances:",retriever.results['distances'])
            print("indices:",indices)
            print("vals:")

            indices = indices.squeeze()
            if indices.dim() == 0:
                #continue
                indices = indices.unsqueeze(0)
            for item in indices:
                if item.item() == vals_np[tag]:
                    print(vals_np[tag])
                    k_dic[tag].append(kl)
                tag += 1
    for i in range(100):
        print(k_dic[i])
        
            

        #m = max(knn_prob)
    print("OK")
    tf = open(os.path.join(ds_path,"select_k_for_all.json"), "w")
    json.dump(k_dic,tf)
    tf.close()
    print("已将收集数据保存至目录。///")

    # for i,item in enumerate(datastore['keys'].data):
    #     for k in klist:
    #         retriever.retrieve(item, return_list=["vals", "distances"],k = k)
    #         knn_prob =combiner.get_knn_prob(retriever.results, device=log_output[0].device)
    #         m = max(knn_prob)
    #         if(knn_prob.index(m) == vals_np[i]):
    #             ktag.append(k)
    #     kcount = Counter(ktag)
    #     ksort = list(kcount).sorted(reversed = True)
    #     kfinal.append(ksort[0])
    # tf = open(os.path.join(ds_path,"select_k.json"), "w")
    # json.dump(kfinal,tf)
    # tf.close()
    # print("已将收集数据保存至目录。///")


    print("加载数据...")

    return 0

def get_CF(ds_path):

    datastore = Datastore.load(f"{ds_path}", load_list=["keys","vals"])
    print("加载数据...")
    vals_np= np.array(datastore['vals'].data, dtype=np.int64)
    # first_ten_items = vals_np[:10]
    # for idx, item in enumerate(first_ten_items):
    #     print(f"Element {idx}: {item}")
    count = Counter(vals_np)
    count = {int(k): v for k, v in count.items()}
    for key in islice(count.keys(), 10):
        print(f"key:{key},num:{count[key]}")
    tf = open(os.path.join(ds_path,"dictionary.json"), "w")
    json.dump(count,tf)
    tf.close()
    print("已将收集数据保存至目录。///")

##parser 被创建并配置为用于解析 Fairseq 框架中验证过程的命令行参数。它包括与数据集、分布式训练、模型和评估相关的参数。这个 parser 对象最终被返回，以便在其他部分的代码中使用
## knnbox code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def get_build_datastore_parser(default_task=None):
    r"""
    very similar to options.get_validation_parser() but parse arch as well.

    Difference:
    - when validate, we don't need to specify --arch and model args, because they are
    recorded in .pt file.

    - when building datastore, we need to load the saved model parameter to a knn-mt arch,
    which is different from the checkpoint original arch.
    For example, I have a nmt checkpoint with arch `transformer_iwslt_de_en`, and now I want to
    load it's parameter to arch `vanilla@transformer_iwslt_de_en`, I must specify
    arch = "vanilla@transfromer_iwslt_de_en".
    """
    parser = options.get_parser("Validation", default_task)
    options.add_dataset_args(parser, train=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    # knnbox add one line below to parse arch
    options.add_model_args(parser)
    group = parser.add_argument_group("Evaluation")
    from fairseq.dataclass.data_class import CommonEvalParams
    options.gen_parser_from_dataclass(group, CommonEvalParams())
    return parser
## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end 

def cli_main():
    ## knnbox related code start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # parser = options.get_validation_parser()
    parser = get_build_datastore_parser()
    args = options.parse_args_and_arch(parser)


    ## only override args that are explicitly given on the command line
    # override_parser = options.get_validation_parser()
    override_parser = get_build_datastore_parser()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)




if __name__ == "__main__":
    cli_main()
    

