from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
import typing
from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner
import torch.nn as nn
import torch

### Added for measuring KNN time
from fairseq.logging.meters import StopwatchMeter
import torch

def record_timer_start(t : StopwatchMeter):
    torch.cuda.synchronize()  # Synchronize to ensure precision.
    t.start()
    
def record_timer_end(t : StopwatchMeter):
    torch.cuda.synchronize()  # Synchronize to ensure precision.
    t.stop()

# class DistanceAwareRepresentations:
#     def __init__(self, input_size, hidden_sizes : typing.Union[int, typing.List[int]], output_size) -> None:
#         self.input_size = input_size
#         self.output_size = output_size
#         inner_modules = []
        
#         for hi, hs in enumerate(hidden_sizes):
            
class DistanceAwareRepresentations(nn.Module):
    def __init__(self, input_size, hidden_size : typing.Union[int, typing.List[int]], output_size) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.l1 = nn.Linear(input_size, hidden_size - input_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.l1(X))
        return self.l2(torch.cat([X, z], dim=-1))        
    
@register_model("BNR_knn_mt")
class BNRKNNMT(TransformerModel):
    r"""
    The BNR knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "train", "inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of BNR knn-mt")
        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
                            help="The hyper-parameter lambda of BNR knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of BNR knn-mt")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with BNRKNNMTDecoder
        """
        return BNRKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


    def after_inference_hook(self):
        if hasattr(self.decoder, "after_inference_hook"):
            self.decoder.after_inference_hook()

class BNRKNNMTDecoder(TransformerDecoder):
    r"""
    The BNR knn-mt Decoder, equipped with knn datastore, retriever and combiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        if args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another 
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)  
            self.datastore = global_vars()["datastore"]

        elif args.knn_mode == "inference":
            # when inference, we don't load the keys, use its faiss index is enough
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda,
                     temperature=args.knn_temperature, probability_dim=len(dictionary))
            
            #self.retrieve_timer = StopwatchMeter()
        elif args.knn_mode == 'train':
            self.datastore = Datastore.load(args.knn_datastore_path, load_list=["vals"])
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_k)
            self.combiner = Combiner(lambda_=args.knn_lambda,
                     temperature=args.knn_temperature, probability_dim=len(dictionary))
            self.distance_aware_representations = DistanceAwareRepresentations(self.output_projection.in_features, self.output_projection.in_features * 3, self.output_projection.in_features)

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[EncoderOut] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.args.knn_mode == "build_datastore":
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            # save half precision keys
            self.datastore["keys"].add(keys.half())
        
        elif self.args.knn_mode == "inference":
            ## query with x (x needn't to be half precision), 
            ## save retrieved `vals` and `distances`
            #record_timer_start(self.retrieve_timer)
            self.retriever.retrieve(x, return_list=["vals", "distances"])
            #record_timer_end(self.retrieve_timer)
        
        elif self.args.knn_mode == 'train':
            new_representations = self.distance_aware_representations(x)
            self.retriever.retrieve(x, return_list=["keys", "vals", "distances"])
            results = self.retriever.results
        
        if not features_only:
            x = self.output_layer(x)
        return x, extra
    

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieve resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == "inference":
            #record_timer_start(self.retrieve_timer)
            knn_prob = self.combiner.get_knn_prob(**self.retriever.results, device=net_output[0].device)
            combined_prob, _ = self.combiner.get_combined_prob(knn_prob, net_output[0], log_probs=log_probs)
            #record_timer_end(self.retrieve_timer)
            return combined_prob
        else:
            return super().get_normalized_probs(net_output, log_probs, sample)
        
    def after_inference_hook(self):
        if self.retrieve_timer.start_time is None:
            print("KNN overhead time is not recoreded")
        else:
            print(f"KNN overhead time = {self.retrieve_timer.sum}s")


r""" Define some BNR knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

@register_model_architecture("BNR_knn_mt", "BNR_knn_mt@transformer_zh_en")
def transformer_zh_en(args):
    archs.transformer_zh_en(args)
    

        

