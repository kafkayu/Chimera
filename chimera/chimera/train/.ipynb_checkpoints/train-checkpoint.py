# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# Adapted from: https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train.py


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
from transformers.trainer import *
from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template
from torch.nn import CrossEntropyLoss,MSELoss
from torch.nn import functional as F
import os

from medusa.model.medusa_model import MedusaModel, MedusaConfig,SingleMedusa
import torch.nn.functional as F
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# Customized for training Medusa heads
class CustomizedTrainer(Trainer):
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[:][:]
                        
                    #import pdb;pdb.set_trace()   
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]
        logits = torch.transpose(logits, 0, 1).contiguous()
        #logits =logits.unsqueeze(0)
        #import pdb;pdb.set_trace()
        return (loss, logits, labels)
    # def save_model(self, output_dir=None, _internal_call=False):
    #     # import pdb;pdb.set_trace()
    #     # output_dir = self.args.output_dir
    #     # 创建输出目录
    #     os.makedirs(output_dir, exist_ok=True)
 
    #     # 保存训练参数
    #     torch.save(
    #     self.model.trimlp.state_dict(),
    #     os.path.join(output_dir, "medusa_lm_head.pt"),
    #   )
    #     torch.save(self.model.fast_layer1.state_dict(), os.path.join(output_dir, "fast_layer1.pt"))
    #     torch.save(self.fastoutput.state_dict(),os.path.join(output_dir, "fastouput.pt"))
    #     # # 保存有梯度变化的模型参数
    #     # saved_params = {
    #     #     k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
    #     # }
    #     torch.save(self.model.medusa_head.state_dict(), os.path.join(output_dir, "medusa_head.pt"))
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # DDP will give us model.module
        if hasattr(model, "module"):
            medusa = model.module.medusa
        else:
            medusa = model.medusa
    
        logits1 = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        logits =logits1["logits"]
        
        labels = inputs["labels"]
        labels2 = torch.argmax(logits1["logits"][-1],dim=-1)
        #import pdb;pdb.set_trace()
        # Shift so that tokens < n predict n
        loss = 0
        loss_fct =CrossEntropyLoss()
        log = {}
        #logits = torch.clamp(logits, min=1e-7, max=100 - 1e-7)
        for i in range(medusa):
            
            medusa_logits = logits[i, :, 2:-2 ].contiguous()
            
            medusa_labels = labels[...,  4:].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = medusa_labels.view(-1)        
            medusa_labels = medusa_labels.to(medusa_logits.device)      
            loss_i = loss_fct(medusa_logits, medusa_labels)
            #loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]

            # Add top-k accuracy
            for k in range(1, 6):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_top{k}"] = correct.float().mean().item()
            log[f"medusa{i}_loss"] = loss_i.item()

            medusa_logits = logits[i, :, 2:-1 ].contiguous()
            medusa_logits = medusa_logits.view(-1, logits.shape[-1])
            medusa_labels = labels2[...,  3:].contiguous()
            medusa_labels = medusa_labels.view(-1)
            medusa_labels = medusa_labels.to(medusa_logits.device)
            loss_i = loss_fct(medusa_logits, medusa_labels)
            loss += loss_i
            not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
            medusa_labels = medusa_labels[not_ignore]
            for k in range(1, 6):
                _, topk = medusa_logits.topk(k, dim=-1)
                topk = topk[not_ignore]
                correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
                log[f"medusa{i}_model_top{k}"] = correct.float().mean().item()
            log[f"medusa{i}_model_loss"] = loss_i.item()
 
        

        
       
        self.log(log)
        return (loss+logits1['hsloss'], logits1["logits"]) if return_outputs else loss+logits1['hsloss']
   

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.3")
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Load in 4 bit."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load in 8 bit."},
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="sharegpt_clean.json",
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    medusa_num_heads: int = field(
        default=1,
        metadata={"help": "Number of Medusa heads."},
    )
    medusa_num_layers: int = field(
        default=1,
        metadata={"help": "Number of layers for each Medusa head."},
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}, {j}, {role}, {conv.roles[j % 2]}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets. Only compute loss on the assistant outputs.
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)

            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the LLaMA tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def compute_metrics(pred):
        logits,labels = pred
        loss = 0
        log = {}
        loss_fct = CrossEntropyLoss()

        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        labels2 = torch.argmax(logits[ :,-1],dim=-1)

        medusa_logits = logits[ :,0,2:-2  ].contiguous()   
        medusa_labels = labels[...,4:].contiguous()
        medusa_logits = medusa_logits.view(-1, logits.shape[-1])
        medusa_labels = medusa_labels.view(-1)
        
        medusa_labels = medusa_labels.to(medusa_logits.device)
        #import pdb;pdb.set_trace()
        #medusa_logits = torch.clamp(medusa_logits, min=1e-7, max=1 - 1e-7)
        #import pdb;pdb.set_trace()
        loss_i = loss_fct(medusa_logits, medusa_labels)
        loss += loss_i
        not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
        medusa_labels = medusa_labels[not_ignore]

        # Add top-k accuracy
        for k in range(1, 6):
            _, topk = medusa_logits.topk(k, dim=-1)
            topk = topk[not_ignore]
            correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
            log[f"eval_medusa{0}_top{k}"] = correct.float().mean().item()    
        log[f"eval_medusa{0}_loss"] = loss_i.item()
        ###########model_prediction
        medusa_logits = logits[ :,0,2:-1  ].contiguous()  
        medusa_logits = medusa_logits.view(-1, logits.shape[-1])
        medusa_labels = labels2[...,  3:].contiguous()
        medusa_labels = medusa_labels.view(-1)
        medusa_labels = medusa_labels.to(medusa_logits.device)
        loss_i = loss_fct(medusa_logits, medusa_labels)
        not_ignore = medusa_labels.ne(IGNORE_TOKEN_ID)
        medusa_labels = medusa_labels[not_ignore]
        for k in range(1, 6):
            _, topk = medusa_logits.topk(k, dim=-1)
            topk = topk[not_ignore]
            correct = topk.eq(medusa_labels.unsqueeze(-1)).any(-1)
            log[f"eval_medusa{0}_model_top{k}"] = correct.float().mean().item()
        log[f"eval_medusa{0}_model_loss"] = loss_i.item()


        return log
def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    print(ModelArguments)
    print(ModelArguments)
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank =0 #training_args.local_rank
    
    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if model_args.load_in_4bit else None,
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
    )

    #Freeze the base model


    ###########加载transformer############
    import copy
    #fast_Layer = copy.deepcopy(model.model.layers[-1])
    
    #############********加载旧模型头*******###########
    
    
    # Add Medusa heads
    medusa_lm_head = MedusaModel(
        model,
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path
    )
    #######load pretrained model####
    state_name = 'idea12_2fastlayer_0108_medusa_mlp_vicuna-7b-v1.3_medusa_1_lr_0.0001_layers_1/checkpoint-7500/pytorch_model.bin'
    dict =torch.load(state_name)
    medusa_lm_head.load_state_dict(dict)
    del dict
    torch.cuda.empty_cache()#清除无用变量
    ########
    for param in medusa_lm_head.base_model.parameters():
        param.require_grad = False

    training_args.output_dir = f"{training_args.output_dir}_medusa_mlp_{model_args.model_name_or_path.split('/')[-1]}_medusa_{training_args.medusa_num_heads}_lr_{training_args.learning_rate}_layers_{training_args.medusa_num_layers}"

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Generate Medusa config for pushing to HF hub
    medusa_config = MedusaConfig(
        medusa_num_heads=training_args.medusa_num_heads,
        medusa_num_layers=training_args.medusa_num_layers,
        base_model_name_or_path=model_args.model_name_or_path,
    )


    #######加载checkpoint#####
    
    
    # Save Medusa config
    medusa_config.save_pretrained(training_args.output_dir)

    # import pdb; pdb.set_trace()
    # Start trainner
    trainer = CustomizedTrainer(
        model=medusa_lm_head, tokenizer=tokenizer, args=training_args,compute_metrics=compute_metrics, **data_module
    )
    if hasattr(medusa_lm_head, "module"):
        lm_head = medusa_lm_head.module.medusa_head
    else:
        lm_head = medusa_lm_head.medusa_head

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    model.config.use_cache = True
    # trainer.save_state()
    # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    # Save MedusaHead seperately
    if hasattr(medusa_lm_head, "module"):
        lm_head = medusa_lm_head.module.medusa_head
    else:
        lm_head = medusa_lm_head.medusa_head

    # Save Medusa heads
    torch.save(
        lm_head.state_dict(),
        os.path.join(training_args.output_dir, "medusa_lm_head.pt"),
    )

    torch.save(
        medusa_lm_head.trimlp.state_dict(),
        os.path.join(training_args.output_dir, "trimlp.pt"),
    )
    torch.save(
        medusa_lm_head.fast_layer1.state_dict(),
        os.path.join(training_args.output_dir, "fastlayer1.pt"),
    )

    #)
    torch.save(
            medusa_lm_head.fast_layer0.state_dict(),
            os.path.join(training_args.output_dir, "fastlayer0.pt"),
        )



if __name__ == "__main__":
    train()
