from typing import Any, Dict, List, Optional, Tuple
from torch import Tensor
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
import os
import math
import random
import torch.nn.functional as F
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from collections import Counter
from knnbox.common_utils import global_vars, select_keys_with_pad_mask, archs
from knnbox.datastore import Datastore
from knnbox.retriever import Retriever
from knnbox.combiner import Combiner

### Added for measuring KNN time
from fairseq.logging.meters import StopwatchMeter
import torch
import json
def record_timer_start(t : StopwatchMeter):
    torch.cuda.synchronize()  # Synchronize to ensure precision.
    t.start()
    
def record_timer_end(t : StopwatchMeter):
    torch.cuda.synchronize()  # Synchronize to ensure precision.
    t.stop()


@register_model("vanilla_knn_mt")
class VanillaKNNMT(TransformerModel):
    r"""
    The vanilla knn-mt model.
    """
    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices= ["build_datastore", "inference"],
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
    
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with VanillaKNNMTDecoder
        """
        return VanillaKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )


    def after_inference_hook(self):
        if hasattr(self.decoder, "after_inference_hook"):
            self.decoder.after_inference_hook()

class VanillaKNNMTDecoder(TransformerDecoder):
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
                     temperature=args.knn_temperature, probability_dim=len(dictionary),datastore_path=args.knn_datastore_path)
            
            self.retrieve_timer = StopwatchMeter()
            self.select_data = {}
            self.k_parameters = {}
            # with open(os.path.join(self.args.knn_datastore_path,"select_k.json"), 'r') as f1:
            #     self.select_data = json.load(f1)
            # with open(os.path.join(self.args.knn_datastore_path,"select_k_2.json"), 'r') as f3:
            #     self.select_data_2 = json.load(f3)
            # with open(os.path.join(self.args.knn_datastore_path,"parameters.json"), 'r') as f2:
            #     self.k_parameters = json.load(f2)

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

        ###x储存hidden_state
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
            #检索器返回最近的几个点的val和distance
            ###这里先对xretrive一次，返回keys和values；然后遍历所有的keys，每个keyi遍历一遍k值，进行retrive(keyi,k),返回values和distances，利用combiner计算一下结果，如果结果正确那么把k扔进klist里面。最后统计一下klist里面最大数量的值，作为真正的key值，进行下面的步骤。
            q1 = []
            q2 = []
            # with open(os.path.join('/data/qirui/z-testdata','example7.txt'), 'a') as file:
            #                     string = "first:"+ str(x.size())+str(x.cpu().numpy())+ " "
            #                     file.write(string)
            #以下是修改的部分
            select_v = []
            segma = 1
            Rev = False
            nl = [3,5,7,8,9,11,6,10,13,15]
            mytaglist =[3,5,7,9,11] #控制k的种类
            dmin = self.k_parameters["dmin"]
            dmax = self.k_parameters["dmax"]
            vmax = self.k_parameters["vmax"]
            # with open(os.path.join('/data/qirui/z-testdata','example2_5.txt'), 'a') as file:
            #         string = "first:"+ str(x.size())+"\n"
            #         file.write(string)
            self.retriever.retrieve(x, return_list=["distances","indices","vals"] ,k = math.ceil(vmax/segma))
            # with open(os.path.join('/data/qirui/z-testdata','test1.txt'), 'a') as file:
            #         string = "first:"+ str(self.retriever.results["distances"].size())+ " "
            #         file.write(string)
            d1nn = self.retriever.results["distances"][:,:,0]#可能是[batchsize,1,1]
            # with open(os.path.join('/data/qirui/z-testdata','example2_6.txt'), 'a') as file:
            #         string = "first:"+ str(d1nn.size())+ "\n"
            #         file.write(string)
            
            d1nn = d1nn.cpu().numpy()#.view(-1)
            target_size = 15
            
            #看一下d1nn的区别
            for item in d1nn:
                it = item.item()
                with open(os.path.join('/data/qirui/z-testdata','exampled_fast.txt'), 'a') as file:
                    string =  str(it)+ "\n"
                    file.write(string)
                if it < dmin:
                    select_v.append(vmax / segma)
                    #select = vmax / segma#
                elif it > dmax:
                    select_v.append(1)
                    #select = 1#改的不是这个！
                else:
                    beta = -math.log(vmax)/(dmax-dmin)
                    vlin = (1-vmax)/(dmax-dmin)*(it-dmin)+vmax
                    vexp = vmax * math.exp((it-dmin)*beta)
                    vyi = math.ceil(((vlin * vexp) ** 0.5)/ segma)
                    #vyi = math.ceil(((vlin + vexp) * 0.5)/ segma)
                    #vyi = math.ceil((2 / (1 / vlin + 1 / vexp) )/ segma)
                    ###
                    # if vyi > 8 :
                    #     vyi = 8
                    select_v.append(vyi)##可能是batchsize个数字
                    #select = vyi
            
            for it,select in enumerate(select_v):#x:[bs,1,1024]
                #select = select_v[it]
                index2 = self.retriever.results["indices"][it,:,:select]#(batch_size,select)
                #index2表示这个向量的select个邻居
                klist = []
                selected_k = 0
                index = index2.cpu()  # 将 tensor 移动到 CPU
                index = index.view(-1)#这里把向量的序号都展开了，batchsize！=1的时候注意一下
                # with open(os.path.join('/data/qirui/z-testdata','exampleb.txt'), 'a') as file:
                #     string = "first:"+ str(len(index))+ " "+str(select)
                #     file.write(string)
                for idx in index:
                    idx_str = str(idx.item())#这里的idx.item()是一个单独的key对应的序号
                    #print(index)
                    #idx = idx.cpu()########
                    
                    for item in self.select_data[idx_str]:
                        if item in mytaglist:  #####
                            klist.append(item) 
                    # for item in self.select_data_2[idx_str]:
                    #     if item in mytaglist:  #####
                    #         klist.append(item) 
                if len(klist) == 0:
                    selected_k = 8
                else:
                    klist_2 = sorted(klist,reverse=Rev)
                    count = Counter(klist_2)
                    #count = Counter(klist)
                    #
                    selected_k = max(count.items(), key=lambda x: x[1])[0]
                q1.append(self.retriever.results["distances"][it,:,:selected_k].unsqueeze(0))#这里时候加的！
                q2.append(self.retriever.results["vals"][it,:,:target_size].unsqueeze(0))
                # with open(os.path.join('/data/qirui/z-testdata','examplec.txt'), 'a') as file:
                #     string = "first:"+ str(self.retriever.results["vals"][it,:,:target_size].size())+ " "
                #     file.write(string)
                # with open(os.path.join('/data/qirui/z-testdata','example5.txt'), 'a') as file:
                #                 string = "first:"+str(selected_k)+" "
                #                 file.write(string)
                # with open(os.path.join('/data/qirui/z-testdata','example6.txt'), 'a') as file:
                #                 string = "first:"+str(count)+" "
                #                 file.write(string)
                # with open(os.path.join('/data/qirui/z-testdata','example.txt'), 'a') as file:
                #                 string = "first:"+str(select)+" "
                #                 file.write(string)
            
            #padding
            

            q1_padded = pad(q1,target_size,999.)#1e9)
            #q2_padded = pad(q2,target_size,random.randint(0, 100))   
            q2_padded = q2
            squeezed_tensors_q1 = [torch.squeeze(tensor,dim = 0) for tensor in q1_padded]
            squeezed_tensors_q2 = [torch.squeeze(tensor,dim = 0) for tensor in q2_padded]
            
            # with open(os.path.join('/data/qirui/z-testdata','examplea.txt'), 'a') as file:
            #         string = "first:"+ str(len(squeezed_tensors_q1))+str(squeezed_tensors_q1)+ " "
            #         file.write(string)
            self.retriever.results["distances"] = torch.cat(squeezed_tensors_q1, dim=0).unsqueeze(1)#.unsqueeze(-1)
            self.retriever.results["vals"]=torch.cat(squeezed_tensors_q2, dim=0).unsqueeze(1)#.unsqueeze(-1)        
            
            
            
                
                
   
            #方法二原始版本
            # index = index2.cpu()  # 将 tensor 移动到 CPU
            # index = index.view(-1)
            # for idx in index:
            #     idx_str = str(idx.item())#这里的idx.item()是一个单独的key对应的序号
            #     #print(index)
            #     #idx = idx.cpu()########
            #     for item in self.select_data[idx_str]:  
            #         klist.append(item)
            #     # for item in self.select_data[idx]:
            #     #      klist.append(item)
            # if len(klist) == 0:
            #     selected_k = 8
            # else:
            #     count = Counter(klist)
            #     selected_k = max(count.items(), key=lambda x: x[1])[0]
            # self.retriever.retrieve(x, return_list=["vals","distances"], k = selected_k)
            # #self.retriever.retrieve(x, return_list=["vals","distances"], k = 9)






            #record_timer_end(self.retrieve_timer)
        
        if not features_only:#？
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
            #用combiner把knn的概率取到，并结合在一起
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
def pad(q,target_size,num):
        q_padded = []
        for tensor in q:
            # 当前张量的最后一个维度长度
            current_size = tensor.size(-1)
            padding_size = target_size - current_size
            if padding_size > 0:
                pad_tensor = F.pad(tensor, (0, padding_size), "constant", num)#0)
            else:
                pad_tensor = tensor
            q_padded.append(pad_tensor)
        return q_padded

r""" Define some vanilla knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""
@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)

# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)

# default parameters used in tensor2tensor implementation
@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)

@register_model_architecture("vanilla_knn_mt", "vanilla_knn_mt@transformer_zh_en")
def transformer_zh_en(args):
    archs.transformer_zh_en(args)
    

        

