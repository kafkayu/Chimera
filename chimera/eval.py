# import sys
# sys.stdout = open('output.txt', 'w')
from chimera.model.Chimera_model import ChimeraModel, ChimeraConfig
from chimera.model.modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
import os
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
from chimera.model.Chimera_model import ChimeraModel, ChimeraConfig
import torch.nn.functional as F
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
model_name_or_path= '../chimera_test_chimera_mlp_vicuna-7b-v1.3_chimera_1_lr_0.0001_layers_1'
chimera_model = ChimeraModel.from_pretrained(
       chimera_name_or_path=model_name_or_path
    )
import json
filename = "../../data/ShareGPT_Vicuna_unfiltered/small_question.json"########训练数据
savefilename = "../../data/ShareGPT_Vicuna_unfiltered/small_question_answer.json"
with open(filename, 'r') as f:
    train = json.load(f)
tokenizer = chimera_model.get_tokenizer()
for i in range(len(train['question'])):
    print("第{}个问题".format(i))
    if i == 2:
        break
    Q = train['question'][i]['value']
    output,ratio = chimera_model.generate(Q,choices=4,max_predictlen = 4,max_length=30,topk=1,greedy_search=False)
    train['question'][i]['answer'] = tokenizer.decode(output[0])
    
    with open( savefilename, "w") as json_file:
        json.dump(train, json_file)


