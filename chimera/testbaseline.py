import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # define GPU id, remove if you want to use all GPUs available
import torch
from tqdm import tqdm
import time
from contextlib import contextmanager
import numpy as np
# from chimera.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
# from chimera.model.Chimera_model import ChimeraModel,ChimeraConfig
# from chimera.model.kv_cache import *
# from chimera.model.utils import *
# from chimera.model.choices import *
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig
# from chimera.model.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import  LlamaModel,LlamaDecoderLayer

import transformers
from transformers import Trainer, BitsAndBytesConfig
import torch
import json
from fastchat.model.model_adapter import get_conversation_template

import sys
print('model',sys.argv[1])
model_name_or_path = sys.argv[1]
name = model_name_or_path.split('/')[-1]
#import pdb;pdb.set_trace()
filename =  ""
print(filename)
import pickle
token_dict = {}
@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

from transformers import BitsAndBytesConfig

model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        use_cache=True,
     torch_dtype=torch.float16,
        load_in_8bit=True,
    
    )

tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        use_fast=False,
    )
tokenizer.pad_token = tokenizer.unk_token
with open( "", "r") as json_file:
   Qs =  json.load( json_file)
#filename =  ""
conv = get_conversation_template("llama-1")
count = 0
for i in range(len(Qs)):
    count=count+1
    if count % 2 == 0:
        with open( filename, "w") as json_file:
            json.dump(Qs,json_file)
        print("{}question".format(count))
    conv.messages = []
    for j in range(len(Qs[i]['conversations'])):
        
        if Qs[i]['conversations'][j]['from']=='human':
            #print("ok")
            
            conv.append_message(conv.roles[0], Qs[i]['conversations'][j]['value'])
            inputs = conv.get_prompt() +conv.roles[1] +': '
            torch.cuda.synchronize()
            start = time.time()
            input = tokenizer([inputs])
            output = model.generate(inputs = torch.tensor(input['input_ids'] ).to("cuda"),max_new_tokens =300)
            torch.cuda.synchronize()
            elp = time.time()-start
            #print(output)
            Qs[i]['conversations'][j+1]['value'] = tokenizer.decode(output[0][len(input['input_ids'][0]):],skip_special_tokens =True)
            
            conv.append_message(conv.roles[1], Qs[i]['conversations'][j+1]['value'])
            Qs[i]['conversations'][j+1]['speed'] = (len(output[0])-len(input['input_ids'][0]))/elp
with open( filename, "w") as json_file:
            json.dump(Qs,json_file)
            print("{}question".format(count))            
