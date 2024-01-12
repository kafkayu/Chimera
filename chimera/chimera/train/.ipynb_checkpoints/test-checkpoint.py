import sys
sys.stdout = open('output.txt', 'w')
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, BitsAndBytesConfig
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import os
from medusa.model.medusa_model import MedusaModel, MedusaConfig,SingleMedusa
import torch.nn.functional as F
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

def medusa_forward(input_ids, model, tokenizer, medusa_buffers, medusa_topk, temperature, posterior_threshold, posterior_alpha, past_key_values, past_key_values_data, current_length_data, steps = 512):
    wall_times = {'medusa': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}
    
    with timed(wall_times, 'init'):
        reset_medusa_mode(model)
        input_len = input_ids.shape[1]
        medusa_logits, logits = initialize_medusa(input_ids, model, medusa_buffers['medusa_attn_mask'], past_key_values)
    
    new_token = 0

    for idx in range(steps): 
        with timed(wall_times, 'medusa'):
            candidates, tree_candidates = generate_candidates(medusa_logits, logits, medusa_topk, medusa_buffers['tree_indices'], temperature)

        with timed(wall_times, 'tree'):
            medusa_logits, logits, outputs = tree_decoding(model, tree_candidates, past_key_values, medusa_buffers['medusa_position_ids'], input_ids, medusa_buffers['retrieve_indices'])

        with timed(wall_times, 'posterior'):
            best_candidate, accept_length = evaluate_posterior(logits, candidates, temperature, posterior_threshold, posterior_alpha)
        
        with timed(wall_times, 'update'):
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(input_ids, candidates, best_candidate, accept_length, medusa_buffers['retrieve_indices'], outputs, logits, medusa_logits, new_token, past_key_values_data, current_length_data)

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break

    return input_ids, new_token, idx, wall_times
state_name = '../../../../idea12_2fastlayer_0108_medusa_mlp_vicuna-7b-v1.3_medusa_1_lr_0.0001_layers_1/checkpoint-7500/pytorch_model.bin'
dict =torch.load(state_name)
model_name_or_path="../../../../../model/vicuna-7b-v1.3"
config = transformers.AutoConfig.from_pretrained(
    model_name_or_path,
)
model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        low_cpu_mem_usage=True,
 
    )
medusa_lm_head = MedusaModel(
        model,
        medusa_num_heads=1,
        medusa_num_layers=1,
        base_model_name_or_path=model_name_or_path
    )
medusa_lm_head.load_state_dict(dict)
model_max_length=1024
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=model_max_length,
    padding_side="right",
    use_fast=False,
)
def fastlayer_forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        last_hs = None,
        orig = None,
        
    ):
        if last_hs is None:
            with torch.inference_mode():
                # Pass input through the base model
                outputs = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    #output_hidden_states=True,
                )
                last_hs = outputs[0][:,:-1]
                #orig = self.base_model.lm_head(outputs[0])
        #####1.get trigram#####
        embed =self.base_model.model.embed_tokens(input_ids)
        embedtrigram = torch.cat((embed[:,:-2],embed[:,1:-1],embed[:,2:]),dim=-1)
        gram0 = torch.cat((embed[:,0],embed[:,0],embed[:,0]),dim=-1).unsqueeze(1)
        gram1 = torch.cat((embed[:,0],embed[:,1],embed[:,1]),dim=-1).unsqueeze(1)
        embedtrigram = torch.cat((gram0,gram1,embedtrigram),dim=-2)
        embedtrigram = self.trimlp(embedtrigram )
        from modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
        batch_size, seq_length = embed.shape[:2]
        attention_mask = _prepare_4d_causal_attention_mask(
                         attention_mask[:,:], (batch_size, seq_length), embed, 0
                    )
        attention_mask  = attention_mask.to(self.base_model.device)
        
        ########1.2 forward融合信息
        for i in self.fast_layer0:
            embedtrigram = i(embedtrigram,attention_mask =attention_mask )
            embedtrigram = embedtrigram[0]
        
        # #####2.构造新的attention_mask,seq_length
        # # ######3.构造positionid
        position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)
        #print(position_ids.shape)
         
        # #####4.构造新的input,计算结果
        embed2 = torch.cat((last_hs,embedtrigram[:,-1].unsqueeze(1)),dim=-2)
        #print(embed2.shape)
        #print(attention_mask.shape)
        # # # ######首先
        attention_mask[:,:,-2,-1] = 0 
        for i in self.fast_layer1:
            embed2 = i(embed2 ,attention_mask = attention_mask )
            embed2 =embed2[0]

        # #######4.2和最后一层拼接进行计算
        output2 = embed2
        #########5.将layerN拼接作为输入预测据说效果更好 大小为seq-1，由于0，1没有trigram，实际上只有2开始有效
        
        #medusa_logits = []
        #for i in range(self.medusa):
        medusa_logits=self.medusa_head[0](embed2[:,-2] )
        
        return medusa_logits,embed2[:,-2]#{"logits":medusa_logits, dim=0,"hs":embed2[:,-2]}

def predictmoretoken(fastmodel,input_ids,attention_mask,outputs,k=5):
    logit,ca = generate(fastmodel,input_ids,attention_mask,outputs,k)
    for i in ca:
        input_ids = torch.cat((input_ids,i.unsqueeze(0).unsqueeze(0)),dim=-1)
        input_ids    
    
def calacc(input,max_length = 100,k=5):
    input = tokenizer([inputs])
    input_ids = torch.tensor(input.input_ids)
    attention_mask = torch.tensor(input.attention_mask)
    count = 0
    for i in range(max_length):
        
        outputs = medusa_lm_head(input_ids ,attention_mask = attention_mask )
        orig =  outputs['logits'][-1]
        t0 = torch.argmax(orig[0][-1])
        _,predictt0 =  outputs['logits'][1][0][-1].topk(5, dim=-1)
        input_ids  = torch.cat((input_ids,t0.unsqueeze(0).unsqueeze(0)),dim=-1)
        attention_mask = torch.cat((attention_mask,torch.tensor([[1]])),dim=-1)
        
        # l,ca = generate(medusa_lm_head,input_ids,attention_mask,outputs,k=k)
        #realt1 =torch.argmax( medusa_lm_head.base_model(input_ids = input_ids)[0][0][-1])
        
        count+=sum(predictt0.eq(t0))
    print(count/max_length)
    return input_ids
inputs ="Please tell me a story about llama:"# "What is the goal of dating?:"
#tokenizer.decode(output[0])
####generate token 
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
def headgenerate(self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        last_hs = None,
        orig = None, 
        max_len = 3,
        choices = 5,
    ):
    """
    循环产生更多的candidate，保证无损的策略，目前是为了保证greedy search,所以需要准确预测模型的top1。
    输入是seq_len 的last_hs,orig,输出是max_len长度的candidate序列，为了方便起见，这里直接用batch进行存储所有序列
    运行流程：
    1.输入orig , 预测i+1 token
    2.拼接input_ids+ i+1
    3.forward 获得新的fastlayer_hs和新的orig
    4.fastlayer_hs和last_hs拼接，获得新的last_hs
    5.拼接orig生成的newtoken到token中形成新的batch
    """
    prebuffer = []
    for i in range(max_len):
        ####3. forward
        orig ,fs_hs  = fastlayer_forward(medusa_lm_head,input_ids=input_ids,attention_mask=attention_mask,last_hs = last_hs)
        #####4.拼接新last_hs

        last_hs = torch.cat((last_hs,fs_hs.unsqueeze(1)),dim=-2)
        last_hs = last_hs.repeat_interleave(choices,dim=0)
        ####5.获取新t0
        _,t0 = orig.topk(k=choices,dim=-1)
        t0 = t0.unsqueeze(-1)
        prebuffer.append(t0)
        ######6.拼接,获得新的batch
        input_ids =  input_ids.unsqueeze(1)
        input_ids = input_ids.repeat_interleave(choices,dim=1)
        input_ids = torch.cat((input_ids,t0),dim=-1)
        #print(input_ids.shape)
        input_ids = input_ids.flatten(0,1)
        ######7.更新att_m
        #import pdb;pdb.set_trace();
        attention_mask = torch.cat((attention_mask,attention_mask[:,0].unsqueeze(1)),dim=-1)
        attention_mask = attention_mask.repeat(choices,1)
    return input_ids,prebuffer
def  helpgetpredict(batch,candidate_logits,fastbuffer,topk=2,choices = 5):
    indices =  0 
    output = torch.tensor([])
    for i in range(len(fastbuffer)):
        ###1.首先取所有接受结果的topK
        _,topK = candidate_logits[indices][i].topk(k=topk,dim=-1)
        ###2.然后将fastbuffer对应位置token和topK进行比较，查看是否有合适的
        mask = torch.isin(fastbuffer[i], topK)
        ##判断，如果失败就直接退出
        
        ###3.更新indices，这里首先确定在fastbuffer的位置，然后由于batch的原因，所以需要更新buffer的位置
        
        if mask.sum() == 0: 
            indices=indices[0]
             
            for j in range(i):
                indices = math.floor(indices/choices)
                output =  torch.cat((fastbuffer[len(fastbuffer)-2-j][indices],output),dim=-1)
            ##将当前logits,选择top1即可 
            lasttoken = topK[0]
            output = torch.cat((output,lasttoken),dim=-1)
            return output  
        indices = torch.nonzero(mask)
        indices = indices*(len(fastbuffer[i])-i)*choices
    indices= indices[0]
    output =  torch.cat((output,fastbuffer[-1][indices]),dim=-1) 
    for j in range(len(fastbuffer)):
            
            indices = math.floor(indices/choices)
            output =  torch.cat((fastbuffer[len(fastbuffer)-2-j][indices],output),dim=-1)
    return output    
def naive_predict(model,input_ids = None ,candidate = None ,attention_mask=None,past_key_values= None,topk=2):
     """candidate 是所有可能token序列片段
     """
     aclength = 0
     totallen = len(candidate[0])
     best_candidate = torch.tensor([])
     for i in candidate:
        
         input = torch.cat((input_ids,i),dim=-1)
         outputs = model.base_model.model(
                        input_ids=input.unsqueeze(0),
                        attention_mask=attention_mask,
                        past_key_values=past_key_values
                    )
         orig = model.base_model.lm_head(outputs[0])
         
         count = 0
         for j in range(totallen):
             _,tk = orig[0][j].topk(k=topk , dim=-1)
             
             if sum(tk.eq(i[j])) : count = count+1
             else: break
         if count > aclength: 
             aclength = count
             best_candidate = i[:aclength]
         if aclength == totallen:
             return best_candidate
     return best_candidate
def generate(input,max_length = 100,choices=3,max_predictlen = 5,topk=3):
    input = tokenizer([inputs])
    input_ids = torch.tensor(input.input_ids)
    attention_mask = torch.tensor(input.attention_mask)
    past_key_values = None
    output = input_ids 
    count = 0 
    last_hs = None
    fastbuffer =None
    alltoken = []
    ##initial
     
    ####
    for i in range(max_length):
        outputs = medusa_lm_head.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )
        #####回退使用无损加速策略,如果是top5就选择输出，就这样  
        if last_hs is None : last_hs = outputs[0]
        else :last_hs = torch.cat((last_hs,outputs[0]),dim=-2)
        orig = medusa_lm_head.base_model.lm_head(outputs[0])        
        past_key_values = outputs['past_key_values']
        input_ids = torch.argmax(orig[0][-1]).unsqueeze(0).unsqueeze(0)
        attention_mask = torch.cat((attention_mask,attention_mask[:,0].unsqueeze(0)),dim=-1)
        output = torch.cat((output,input_ids),dim=-1) 
        #import pdb;pdb.set_trace();
        candidate, fastbuffer=headgenerate(medusa_lm_head,input_ids=output,attention_mask=attention_mask,last_hs = last_hs,max_len =max_predictlen,
                                           choices = choices)
        
        """evaluate candiate"""
        #import pdb;pdb.set_trace();
        attention_mask_ev = torch.cat((attention_mask,attention_mask[:,0:max_predictlen]),dim=-1)
        best_candidate = naive_predict(medusa_lm_head,input_ids = input_ids[0],candidate = candidate[:,len(output[0]):] ,attention_mask=attention_mask_ev,
                                       past_key_values= past_key_values,topk=topk)
            
        """renew kv,atmk , input,output"""    
        print("速度{}".format(len(best_candidate)+1))
        #print(candidate.shape)
        ####这里直接将预测的token放入
        #import pdb;pdb.set_trace();
        if len(best_candidate) > 0:
            input_ids = torch.cat((input_ids,best_candidate.unsqueeze(0)),dim=-1)
            output =  torch.cat((output,best_candidate.unsqueeze(0)),dim=-1)  
            attention_mask = torch.cat((attention_mask,attention_mask[:,0:len(best_candidate)]),dim=-1)
            tokenizer.decode(output[0])


    print(tokenizer.decode(output[0]))
    print("totallen",len(output[0]))
    print((len(output[0])-len(input.input_ids))/max_length)
    
    return output
output = generate(inputs,max_length =100)