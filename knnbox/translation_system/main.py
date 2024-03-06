import torch
#from knnbox.models import VanillaKNNMT
from fairseq import options,tasks
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.sequence_generator import SequenceGenerator

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

def build_dic(dic_path):
    vocab_dict = {}
    with open(dic_path,'r') as file:
            dic_content = file.read()
    for line in dic_content.strip().split('\n'):
        word, word_id = line.split()
        vocab_dict[word] = word_id
    return vocab_dict

def get_words_from_ids(word_ids, vocab_dict):
    id_to_word_dict = {v: k for k, v in vocab_dict.items()}
    words = [id_to_word_dict.get(word_id, "<>") for word_id in word_ids]
    return words


def preprocess(sentence,vocab_dict):
    # tokenizer = AutoTokenizer.from_pretrained("./bert-base-multilingual-uncased")
    # tokens = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    # input_ids = tokens["input_ids"]
    words = sentence.split()
    word_ids = [vocab_dict.get(word, "UNK") for word in words]
    return word_ids
    
    return dic_content

def postprocess(output):
    #什么格式的输出我不道哇
    a = 0
if __name__ == "__main__":
    parser = get_knn_generation_parser()
    args = options.parse_args_and_arch(parser)

    override_parser = get_knn_generation_parser()
    override_args = options.parse_args_and_arch(override_parser)

    task = tasks.setup_task(args)
    model = task.build_model(args)
    base_model = '/data/qirui/KNN-BOX-copy-copy/pretrain-models/wmt20.en-zh/en-zh.pt'
    if override_args is not None:
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, "model_overrides", "{}")))
    else:
        overrides = None
    state = load_checkpoint_to_cpu(base_model, overrides)
    model.load_state_dict(state["model"], strict=False, args=args)
    
    if args.fp16:
        model.half()
    
    model.cuda()
    model.prepare_for_inference_(args)
    tokenizer = task.build_tokenizer(args)
    bpe = task.build_bpe(args)
    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x
    INPUT_IDS = task.source_dictionary.encode_line(
            encode_fn("hello, how are you?"), add_if_not_exist=False
        ).long().cuda().unsqueeze(0)
    
    seq_gen =SequenceGenerator (
            [model],
            task.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=None
            
        )
    sample = {
        "net_input" : {
            "src_tokens": INPUT_IDS,
            "src_lengths": torch.tensor([[INPUT_IDS.numel()]], device = "cuda")
        }

    }
    res = seq_gen.generate([],sample)
    output_ids = res[0][0]['tokens'].cpu()

    translated_text = decode_fn(task.target_dictionary.string(output_ids))

    print(translated_text)
    
