import sys
print(sys.argv[1])
path = sys.argv[1]#
name = path.split('/')[-1]
#import pdb;pdb.set_trace()
filename =  "testouptut{}.json".format(name)
print(filename)
# import pdb;pdb.set_trace()

import os
import torch
from tqdm import tqdm
import time
from contextlib import contextmanager
import numpy as np
from chimera.model.modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from chimera.model.Chimera_model import ChimeraModel,ChimeraConfig
from chimera.model.kv_cache import *
from chimera.model.utils import *
from chimera.model.choices import *
import transformers
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoConfig,BitsAndBytesConfig
from chimera.model.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import  LlamaModel,LlamaDecoderLayer

@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

import copy

def from_pretrained(
        # self,
        chimera_name_or_path,
        **kwargs,
    ):
        """
        Args:
            chimera_head_name_or_path (str): Name or path of the Chimera head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            ChimeraModel: A ChimeraModel instance loaded from the given path.
        """
        chimera_config = ChimeraConfig.from_pretrained(chimera_name_or_path)
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     chimera_config.base_model_name_or_path
        # )
        config = AutoConfig.from_pretrained(chimera_config.base_model_name_or_path)
        config.num_key_value_heads = config.num_attention_heads
        config.pretraining_tp= 1
        config.rope_scaling= None
        base_model = KVLlamaForCausalLM.from_pretrained(
            chimera_config.base_model_name_or_path, **kwargs,config=config
        )
        print("path",chimera_config.base_model_name_or_path)
        self = ChimeraModel(
            base_model,
            chimera_config.chimera_num_heads,
            chimera_config.chimera_num_layers,
            chimera_config.base_model_name_or_path,
            
        )
        ##1.trimlp layer
        chimera_trimlp_path = os.path.join(chimera_name_or_path, "trimlp.pt")
        if os.path.exists(chimera_trimlp_path):
            filename = chimera_trimlp_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "trimlp.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        self.trimlp.load_state_dict(chimera_state_dict, strict=False)
        
        ##3.fast_layer1
        chimera_fast_layer1_path = os.path.join(chimera_name_or_path, "fast_layer1.pt")
        if os.path.exists(chimera_fast_layer1_path):
            filename = chimera_fast_layer1_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "fast_layer1.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        

        self.fast_layer1.load_state_dict(chimera_state_dict, strict=False)
    
        #4.chimera_head
        chimera_head_path = os.path.join(chimera_name_or_path, "chimera_head.pt")
        if os.path.exists(chimera_head_path):
            filename = chimera_head_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "chimera_head.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        self.chimera_head.load_state_dict(chimera_state_dict, strict=False)
        
        return self


chimera_model = from_pretrained(path,
                     torch_dtype=torch.float16,
                                load_in_8bit=True,
                                low_cpu_mem_usage=True,
                device_map="auto"
                               )


import copy
import torch.nn as nn
from transformers.models.llama.modeling_llama import  LlamaModel,LlamaDecoderLayer
chimera_model.chimera_heads = 4
tokenizer = chimera_model.get_tokenizer()
chimera_choices = mc_sim_7b_63

temperature = 1
posterior_threshold = 0.09
posterior_alpha = 0.3

from fastchat.model.model_adapter import get_conversation_template
# conv = get_conversation_template("vicuna")
# roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
#inputs= "What are the main differences between Python and JavaScript programming languages?"
#conv.append_message(conv.roles[0], inputs)
#prompt = conv.get_prompt() +conv.roles[1]+':'
token_dict={}
def chimera_forward(prompt,chimera_model, tokenizer, chimera_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 100,token_dict={}):
    
    input = tokenizer([prompt])
    input_ids = torch.tensor(input.input_ids).to(chimera_model.base_model.device)
    attention_mask = torch.tensor(input.attention_mask).to(chimera_model.base_model.device)
    wall_times = {'chimera': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}
    chimera_model.base_model.model.chimera_mask = None    
    with timed(wall_times, 'init'):
        if hasattr(chimera_model, "chimera_choices") and chimera_model.chimera_choices == chimera_choices:
            # Load the cached chimera buffer
            chimera_buffers = chimera_model.chimera_buffers
        else:
            # Initialize the chimera buffer
            chimera_buffers = generate_chimera_buffers(
                chimera_choices, device=chimera_model.base_model.device
            )
        chimera_model.chimera_buffers = chimera_buffers
        chimera_model.chimera_choices = chimera_choices
    
        # Initialize the past key and value states
        if hasattr(chimera_model , "past_key_values"):
            past_key_values = chimera_model.past_key_values
            past_key_values_data = chimera_model.past_key_values_data
            current_length_data = chimera_model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(chimera_model.base_model)
            chimera_model.past_key_values = past_key_values
            chimera_model.past_key_values_data = past_key_values_data
            chimera_model.current_length_data = current_length_data
    
        input_len = input_ids.shape[1]
        reset_chimera_mode(chimera_model)
        
        chimera_logits, logits ,chimera_kv= initialize_chimera(
                chimera_model ,input_ids,attention_mask,chimera_buffers["chimera_attn_mask"], past_key_values,token_dict
        )
            
        chimera_model.base_model.model.medusa_mask = chimera_model.base_model.model.chimera_mask
        new_token = 0
        torch.cuda.synchronize()
        start_time = time.time()
        #max_steps = 10####test
        for idx in range(max_steps): 
            with timed(wall_times, 'chimera'):
                candidates, tree_candidates = generate_candidates(
                        chimera_logits,
                        logits,
                        chimera_buffers["tree_indices"],
                        chimera_buffers["retrieve_indices"],
                        fast=True
                    )
                
    
            with timed(wall_times, 'tree'):
                 logits, outputs = tree_decoding(
                        chimera_model,
                        tree_candidates,
                        past_key_values,
                        chimera_buffers["chimera_position_ids"],
                        input_ids,
                        chimera_buffers["retrieve_indices"],
                        chimera_buffers
                    )
                #chimera_logits,
                 
            with timed(wall_times, 'posterior'):
                best_candidate, accept_length = evaluate_posterior(
                        logits, candidates, temperature, posterior_threshold, posterior_alpha
                    )
                #print(candidates[0])
                #print(tokenizer.decode(candidates[0]))
                #import pdb;pdb.set_trace()
            with timed(wall_times, 'update'):
                
                input_ids, logits, chimera_logits, new_token,chimera_kv = update_inference_inputs(chimera_model,
                        input_ids,
                        candidates,
                        best_candidate,
                        accept_length,
                        chimera_buffers["retrieve_indices"],
                        outputs,
                        logits,
                        chimera_logits,
                        new_token,
                        past_key_values_data,
                        current_length_data,
                        chimera_kv,
                        token_dict                                                                          
                                                                                       
                    )
                
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            torch.cuda.synchronize()
            total_time=   time.time() -start_time
    return input_ids, new_token, idx, wall_times,total_time
def baseline_forward(prompt,chimera_model, tokenizer, chimera_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 100,token_dict={}):
    
    input = tokenizer([prompt])
    input_ids = torch.tensor(input.input_ids).to("cuda")
    attention_mask = torch.tensor(input.attention_mask).to("cuda")
    # wall_times = {'chimera': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}
    chimera_model.base_model.model.chimera_mask = None    
 
    if hasattr(chimera_model, "chimera_choices") and chimera_model.chimera_choices == chimera_choices:
        # Load the cached chimera buffer
        chimera_buffers = chimera_model.chimera_buffers
    else:
        # Initialize the chimera buffer
        chimera_buffers = generate_chimera_buffers(
            chimera_choices, device=chimera_model.base_model.device
        )
    chimera_model.chimera_buffers = chimera_buffers
    chimera_model.chimera_choices = chimera_choices

    # Initialize the past key and value states
    if hasattr(chimera_model , "past_key_values"):
        past_key_values = chimera_model.past_key_values
        past_key_values_data = chimera_model.past_key_values_data
        current_length_data = chimera_model.current_length_data
        # Reset the past key and value states
        current_length_data.zero_()
    else:
        (
            past_key_values,
            past_key_values_data,
            current_length_data,
        ) = initialize_past_key_values(chimera_model.base_model)
        chimera_model.past_key_values = past_key_values
        chimera_model.past_key_values_data = past_key_values_data
        chimera_model.current_length_data = current_length_data

    
    input_len = input_ids.shape[1]
    reset_chimera_mode(chimera_model)
    outputs =chimera_model.base_model(input_ids, past_key_values = past_key_values, use_cache=True)

    
    torch.cuda.synchronize()
    start_time = time.time()
    new_token = 0
    #max_steps = 10####test
    for idx in range(max_steps): 
        new_token=new_token+1
        input_id = outputs.logits[:, -1:].argmax(dim=-1)
        outputs = chimera_model.base_model(input_id, use_cache=True, past_key_values = past_key_values)
        input_ids = torch.cat([input_ids, input_id], dim=-1)

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token > 1024:
            break
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    return input_ids, new_token, idx, total_time        

import json
with open( "question.json", "r") as json_file:
   Qs =  json.load( json_file)



conv = get_conversation_template("vicuna")
count = 0
averageratio = []
for i in range(len(Qs)):
    count=count+1
    if count % 2 == 0:
        with open( filename, "w") as json_file:
            json.dump(Qs,json_file)
        print("{}question".format(count))
    conv.messages = []
    print("average ratio",sum(averageratio)/len(averageratio))    
    for j in range(len(Qs[i]['conversations'])):
        
        if Qs[i]['conversations'][j]['from']=='human':

            conv.append_message(conv.roles[0], Qs[i]['conversations'][j]['value'])
            prompt = conv.get_prompt() +conv.roles[1] +': '
            torch.cuda.synchronize()
            
            with torch.inference_mode():
                input_ids = tokenizer([prompt]).input_ids
                output_ids, new_token, idx, wall_time,total_time = chimera_forward(
                                prompt,
                                chimera_model,
                                tokenizer,
                                chimera_choices,
                                temperature,
                                posterior_threshold,
                                posterior_alpha,
                                max_steps = 500,
                     token_dict=token_dict
                            )

                
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids, new_token, idx, wall_time,total_time = chimera_forward(
                                prompt,
                                chimera_model,
                                tokenizer,
                                chimera_choices,
                                temperature,
                                posterior_threshold,
                                posterior_alpha,
                                max_steps = 500,
                     token_dict=token_dict
                            )
                torch.cuda.synchronize()
                ori1 = time.time() - start_time
                
                output_ids = output_ids[0][len(input_ids[0]) :]

                speed =  new_token /total_time
                Qs[i]['conversations'][j+1]["Output length"] =  output_ids.size(-1)
                Qs[i]['conversations'][j+1]["Compression ratio"]= float(new_token / (idx+1))
                Qs[i]['conversations'][j+1]["Speed1"]= float(speed)
                Qs[i]['conversations'][j+1]["Speed2"]= float(new_token/ori1)
                print(Qs[i]['conversations'][j+1]["Speed1"])
                ###baselinetest###
                torch.cuda.synchronize()
                start_time = time.time()
                output_ids2, new_token, idx,total_time = baseline_forward(
                                prompt,
                                chimera_model,
                                tokenizer,
                                chimera_choices,
                                temperature,
                                posterior_threshold,
                                posterior_alpha,
                                max_steps = new_token,
                     token_dict=token_dict
                            )
                torch.cuda.synchronize()
                ori2 = time.time() - start_time
                Qs[i]['conversations'][j+1]["orispeed"]= float(new_token/total_time)
                print(Qs[i]['conversations'][j+1]["orispeed"])
                averageratio.append(Qs[i]['conversations'][j+1]["Speed1"]/Qs[i]['conversations'][j+1]["orispeed"])

                Qs[i]['conversations'][j+1]['value'] = tokenizer.decode(
                    output_ids,
                    
                )
                conv.append_message(conv.roles[1], Qs[i]['conversations'][j+1]['value'])

            

       
with open( filename, "w") as json_file:
            json.dump(Qs,json_file)
        



