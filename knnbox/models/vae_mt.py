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

from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner

### Added for measuring KNN time
from fairseq.logging.meters import StopwatchMeter
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Mapping, Union

def record_timer_start(t : StopwatchMeter):
    torch.cuda.synchronize()  # Synchronize to ensure precision.
    t.start()
    
def record_timer_end(t : StopwatchMeter):
    torch.cuda.synchronize()  # Synchronize to ensure precision.
    t.stop()
    
class CrossEmbedding(nn.Module):
    def __init__(self, dense_features, onehot_features, output_features, bias=True):    
        self.dense_proj = nn.Parameter(torch.randn(size=(1,output_features, dense_features)), requires_grad=True)
        self.onehot_embed = nn.Parameter(torch.randn(size=(onehot_features, output_features)), requires_grad=True)
        self.dense_features = dense_features
        self.onehot_features = onehot_features
        self.output_features = output_features
        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(output_features)), requires_grad=True)
        
    def forward(self, dense_features : torch.Tensor, onehot_features : torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            dense_features: [batch_size, dense_features]
            onehot_features: [batch_size]
        '''
        
        batch_shape = dense_features.shape[:-1]
        batch_size = dense_features.numel() // dense_features.shape[-1]
        
        assert dense_features.shape[0] == onehot_features.shape[0]       
 
        X_ = dense_features.view(batch_size, -1)
        
        Y1 = torch.bmm(self.dense_proj.expand(batch_shape,-1,-1), X_[:,:self.dense_features].unsqueeze(-1)).squeeze(-1)
        Y2 = self.onehot_embed[onehot_features]
        
        if self.bias is None:
            return Y1 + Y2
        else:
            return Y1 + Y2 + self.bias
        
class SampleGenerator(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size) -> None:
        super().__init__()
        self.l1 = nn.Linear(latent_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        X = F.relu(self.l1(X))
        return torch.sigmoid(self.l2(X))
        
class VAEForSimilarSimples(nn.Module):
    def __init__(self, input_size, output_size, condition_size, hidden_size, latent_size = None) -> None:
        super().__init__()
        if latent_size is None:
            latent_size = hidden_size
            
       # self.embed = 
        self.proj_mu = nn.Linear(hidden_size, latent_size) 
        self.proj_log_sigma = nn.Linear(hidden_size, latent_size) 
        
        #self.loss_for_reconstructing = nn.MSELoss(reduction='mean')
        #self.loss_for_reconstructing = nn.BCELoss(reduction='mean')
        
        #self.encoder = VAEEncoder(input_size + condition_size, hidden_size, latent_size)
        self.decoder = SampleGenerator(latent_size + condition_size, hidden_size, output_size)
        
        self.device = 'cpu'
        
        self.latent_size = latent_size
        self.condition_size = condition_size
        
    def loss_KLD(self, mu, sigma):
        return torch.sum(sigma - (1 + torch.log(sigma)) + mu**2, dim=-1).mean()

        
    def forward(self, X : Mapping[str, Union[torch.Tensor, int, str, object]]) -> Mapping[str, torch.Tensor]:
        #mu, log_sigma = self.encoder(torch.cat([X['input'], X['condition']], dim=-1))
        
        sigma = torch.exp(log_sigma)
        
        eps = torch.randn_like(sigma)
        
        z = mu + eps * sigma
        
        ret = {
            "features" : self.decoder(torch.cat([z, X['condition']], dim=-1)),
            "mu" : mu,
            "sigma" : sigma
        }     
        if 'target' in X:
            loss = self.loss_for_reconstructing(ret["features"], X["target"]) + self.loss_KLD(mu, sigma)
            ret["loss"] = loss
        
        return ret
    
    def to(self, device):
        self.device = device
        return nn.Module.to(self, device)
    
    
class RepresentationTransform(nn.Module):
    def __init__(self, query_size, hidden_size_ffn, hidden_size_vae, latent_size = None) -> None:
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(query_size, hidden_size_ffn),
            nn.Linear(hidden_size_ffn, query_size)
        )
        self.vae = CVAE(query_size, query_size, query_size, hidden_size_vae, latent_size)
        
    def forwad(self, X : Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        query = X["query"]
        
        transed_query = query + self.ffn(query)
        
        out = self.vae({
            "input" : transed_query,
            "condition" : query,
            "target" : transed_query
        })
        
        
    def decode(self, query):
        pass
        
        


@register_model("vae_mt")
class VAEKNNMT(TransformerModel):
    r"""
    The vanilla knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "inference", "train_phase1", "train_phase2"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR", 
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter k of vanilla knn-mt")
        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
                            help="The hyper-parameter lambda of vanilla knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of vanilla knn-mt")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")
        parser.add_argument("--random_seed", type=int, default=233)
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with VanillaKNNMTDecoder
        """
        return VAEKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


    def after_inference_hook(self):
        if hasattr(self.decoder, "after_inference_hook"):
            self.decoder.after_inference_hook()

class VAEKNNMTDecoder(TransformerDecoder):
    r"""
    The vanilla knn-mt Decoder, equipped with knn datastore, retriever and combiner.
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
            
            self.retrieve_timer = StopwatchMeter()

        elif args.knn_mode == "train_phase1":
            pass
        elif args.knn_mode == "train_phase2":
            pass

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


r""" Define some vanilla knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("vae_mt", "vae_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("vae_mt", "vae_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("vae_mt", "vae_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("vae_mt", "vae_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("vae_mt", "vae_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("vae_mt", "vae_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("vae_mt", "vae_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("vae_mt", "vae_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

@register_model_architecture("vae_mt", "vae_mt@transformer_zh_en")
def transformer_zh_en(args):
    archs.transformer_zh_en(args)
    

        

