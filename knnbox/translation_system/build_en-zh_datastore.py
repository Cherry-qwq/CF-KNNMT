import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
#from knnbox.models import VanillaKNNMT
from fairseq import options,tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.sequence_generator import SequenceGenerator
import sys

def get_knn_generation_parser(interactive=False, default_task="translation"):
    parser = options.get_parser("Generation", default_task)
    options.add_dataset_args(parser, gen=True)
    options.add_distributed_training_args(parser, default_world_size=1)
    ## knnbox related code >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # compared to options.get_generation_parser(..), knnbox only add one line code below 
    options.add_model_args(parser)
    ## <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< end
    options.add_generation_args(parser)
    if interactive:
        options.add_interactive_args(parser)
    return parser


if __name__ == "__main__":
    src_tokens = []
    tag_tokens = []

    with open(os.path.join("/data/qirui/KNN-BOX-copy-copy/data-bin/en-zh",'cmn.txt')) as file:
        lines = file.readlines()
        print(lines[0])
        for l in lines:
            l = l.strip().split('\t')
            src_tokens.append(l[0])
            tag_tokens.append(l[1])
    with open(os.path.join("/data/qirui/KNN-BOX-copy-copy/data-bin/en-zh",'train.en-zh.en'),'w') as file:
        file.write(
            "\n".join(src_tokens)
        )
    with open(os.path.join("/data/qirui/KNN-BOX-copy-copy/data-bin/en-zh",'train.en-zh.zh'),'w') as file:
        file.write(
            "\n".join(tag_tokens)
        )

                                   

    # parser = get_knn_generation_parser()
    # args = options.parse_args_and_arch(parser)

    # override_parser = get_knn_generation_parser()
    # override_args = options.parse_args_and_arch(override_parser)

    # task = tasks.setup_task(args)
    # model = task.build_model(args)
    # base_model = '/data/qirui/KNN-BOX-copy-copy/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt'
    # if override_args is not None:
    #     overrides = vars(override_args)
    #     overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    # else:
    #     overrides = None
    # state = load_checkpoint_to_cpu(base_model, overrides)
    # model.load_state_dict(state["model"], strict=False, args=args)

    # if args.fp16:
    #     model.half()
    
    # model.cuda()
    # model.prepare_for_inference_(args)
    # tokenizer = task.build_tokenizer(args)
    # bpe = task.build_bpe(args)
    # def encode_fn(x):
    #     if tokenizer is not None:
    #         x = tokenizer.encode(x)
    #     if bpe is not None:
    #         x = bpe.encode(x)
    #     return x

    # def decode_fn(x):
    #     if bpe is not None:
    #         x = bpe.decode(x)
    #     if tokenizer is not None:
    #         x = tokenizer.decode(x)
    #     return x
    # INPUT_IDS = task.source_dictionary.encode_line(
    #         encode_fn(input_text), add_if_not_exist=False
    #     ).long().cuda().unsqueeze(0)
    
    # seq_gen =SequenceGenerator (
    #         [model],
    #         task.target_dictionary,
    #         beam_size=getattr(args, "beam", 5),
    #         max_len_a=getattr(args, "max_len_a", 0),
    #         max_len_b=getattr(args, "max_len_b", 200),
    #         min_len=getattr(args, "min_len", 1),
    #         normalize_scores=(not getattr(args, "unnormalized", False)),
    #         len_penalty=getattr(args, "lenpen", 1),
    #         unk_penalty=getattr(args, "unkpen", 0),
    #         temperature=getattr(args, "temperature", 1.0),
    #         match_source_len=getattr(args, "match_source_len", False),
    #         no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
    #         search_strategy=None
            
    #     )
    # sample = {
    #     "net_input" : {
    #         "src_tokens": INPUT_IDS,
    #         "src_lengths": torch.tensor([[INPUT_IDS.numel()]], device = "cuda")
    #     }

    # }
    # res = seq_gen.generate([],sample)