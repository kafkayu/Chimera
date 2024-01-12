import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import AutoTokenizer
from .utils import *
from .kv_cache import initialize_past_key_values
import os
from huggingface_hub import hf_hub_download
import copy
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import  LlamaModel,LlamaDecoderLayer
from .choices import *
from .modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
import wandb
wandb.login(key="6224ac7517be176065dbe00432983a2ef90fa010")#######wandb key
       
class ChimeraConfig(PretrainedConfig):
    def __init__(
        self,
        chimera_num_heads=1,##this head is aimed at predicting draft
        chimera_num_layers=1,## the number of mlp layers in the head
        exit_early = False,## the original hiddenstates used in the chimera model
        exit_layer = -1,
        base_model_name_or_path='../model/vicuna-7b-v1.3',#"lmsys/vicuna-7b-v1.3"
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.chimera_num_heads = chimera_num_heads
        self.chimera_num_layers = chimera_num_layers
        self.base_model_name_or_path = base_model_name_or_path


class ResBlock(nn.Module):
    """A Residual Block module.

    This module performs a linear transformation followed by a SiLU activation,
    and then adds the result to the original input, creating a residual connection.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()

    def forward(self, x):
        """Forward pass of the ResBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after the residual connection and activation.
        """
        return x + self.act(self.linear(x))


    
class ChimeraModel(nn.Module):
    """The Chimera Language Model .

    This module creates a  2mlp and 2 transformers basically(based on the 'chimera' parameter)
    
    """

    def __init__(
        self,
        base_model,
        chimera_num_heads=1,
        chimera_num_layers=1,
        base_model_name_or_path='../../../model/vicuna-7b-v1.3',#"lmsys/vicuna-7b-v1.3",'../model/vicuna-7b-v1.3',#'../../../../model/vicuna-7b-v1.3'
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            chimera_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            chimera_num_layers (int, optional): Number of ResBlock layers for each Chimera head. Defaults to 0.
        """
        """
        Model:
        trimlp : calculate the trigram of all the token which can be equivalent to shallow transformer in the llm
        fast_layer0 : one layer transformer which can easily build the long distance of the seqence
        fast_layer1 : one layer transformer , fuse the output of llm and the fast_layer0 and output the better hiddenstates of the new generated token
        chimera_head : mlp , output the distribution of the token
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.chimera_heads = chimera_num_heads
        self.chimera_head_num_layers = chimera_num_layers
        self.base_model_name_or_path =base_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Chimera heads
        self.chimera_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * chimera_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(chimera_num_heads)
            ]
        )
        
        import copy

        self.trimlp = nn.Sequential(
                    *([ResBlock(self.hidden_size*3)] ),
                    nn.Linear(self.hidden_size*3, self.hidden_size, bias=False),
                )


        self.fast_layer0 = nn.Sequential(
                                        copy.deepcopy(base_model.model.layers[-1]),                                    
                                        )

        self.fast_layer1 = nn.Sequential(

                                         copy.deepcopy(base_model.model.layers[-1]),
                                         
                                        )

        self.fast_layer0_kv = None
        self.fast_layer1_kv = None
        for param in self.fast_layer0.parameters():
            param.require_grad = True
        for param in self.fast_layer1.parameters():
            param.require_grad = True
        self.chimera_head.to(self.base_model.dtype).to(self.base_model.device)
        self.fast_layer1.to(self.base_model.dtype).to(self.base_model.device)
        self.fast_layer0.to(self.base_model.dtype).to(self.base_model.device)
        self.trimlp.to(self.base_model.dtype).to(self.base_model.device)

        for i in range(chimera_num_heads):
            # Initialize the weights of each chimera_head using the base model's weights
            self.chimera_head[i][-1].weight.data[:]=base_model.lm_head.weight.data
            
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
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
        base_model = KVLlamaForCausalLM.from_pretrained(
            chimera_config.base_model_name_or_path, **kwargs
        )
        print("path",chimera_config.base_model_name_or_path)
        model = cls(
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
        model.trimlp.load_state_dict(chimera_state_dict, strict=False)
        
        ##2.fast_layer0
        chimera_fast_layer0_path = os.path.join(chimera_name_or_path, "fast_layer0.pt")
        if os.path.exists(chimera_fast_layer0_path):
            filename = chimera_fast_layer0_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "fast_layer0.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        model.fast_layer0.load_state_dict(chimera_state_dict, strict=False)
        
        ##3.fast_layer1
        chimera_fast_layer1_path = os.path.join(chimera_name_or_path, "fast_layer1.pt")
        if os.path.exists(chimera_fast_layer1_path):
            filename = chimera_fast_layer1_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "fast_layer1.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        model.fast_layer1.load_state_dict(chimera_state_dict, strict=False)
        
        ##4.chimera_head
        chimera_head_path = os.path.join(chimera_name_or_path, "chimera_lm_head.pt")
        if os.path.exists(chimera_head_path):
            filename = chimera_head_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "chimera_lm_head.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        model.chimera_head.load_state_dict(chimera_state_dict, strict=False)
        
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        output_hidden_states = False,
        
    ):
        """Forward pass of the ChimeraModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Chimera heads.
            (Optional) Original predictions from the base model's LM head.
        """
        
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                output_hidden_states=output_hidden_states,
            )
            orig = self.base_model.lm_head(outputs[0])
        #####1. predict next token
        t0 = torch.argmax(orig[:,-1],dim=-1).unsqueeze(0).T
        input_ids = torch.cat((input_ids,t0),dim=-1)
        #####2.get trigram#####
        embed =self.base_model.model.embed_tokens(input_ids)
        embedtrigram = torch.cat((embed[:,:-2],embed[:,1:-1],embed[:,2:]),dim=-1)
        gram0 = torch.cat((embed[:,0],embed[:,0],embed[:,0]),dim=-1).unsqueeze(1)
        gram1 = torch.cat((embed[:,0],embed[:,1],embed[:,1]),dim=-1).unsqueeze(1)
        embedtrigram = torch.cat((gram0,gram1,embedtrigram),dim=-2)
        embedtrigram = self.trimlp(embedtrigram )
        attention_mask = torch.cat((attention_mask ,attention_mask [:,0].unsqueeze(0).T),dim=-1)
        
        batch_size, seq_length = embed.shape[:2]
        attention_mask = _prepare_4d_causal_attention_mask(
                         attention_mask[:,:], (batch_size, seq_length), embed, 0
                    )
        attention_mask  = attention_mask.to(self.base_model.device)
        ########3. fuse information build the long distance of the seq basically
        for i in self.fast_layer0:
            embedtrigram = i(embedtrigram,attention_mask =attention_mask )
            embedtrigram = embedtrigram[0]
        #######4.build the new attention_mask,because the input is original layerN and  embedtrigram(next token) 
        #######the length of embedtrigram  is seq_length，the length of input is 2seq_length-2
        """the theroy is as follows：
            update the  attentionmask and positionid setting and let the embedtrigram serve as the i+1 layerN-1 which is uncalcualted by the original llm
            and output i+1 fastlayerN
            additionally , teacher and student between  all the i+1 fastlayerN and the i+1 layerN of original model
            finally, we train a chimera head and transform the fastlayerN into the logtis of draft, teacher and student between  all the i+1 fastlayerN logtis and  the i+1 layerN logtis of original model.
            the chimera model can predict the layerN hidden_states and logits of the i+1 layerN-1 which is uncalcualted by the original llm
            repeat this steps and get the lots of draft
            
        """
        attention_mask2 =torch.full((seq_length-1, seq_length-1), -3.4028e+38) + torch.diag(torch.zeros(seq_length-1)+3.4028e+38-1)
        attention_mask2 = attention_mask2.to(self.base_model.device)
        attention_mask3 = torch.cat((attention_mask[0,0,:-1,:-1],attention_mask2[:,:]),dim=-1)
        attention_mask3 = torch.cat((attention_mask3[:,:],attention_mask3[:,:]),dim=-2).unsqueeze(0).unsqueeze(0)
        attention_mask3 = attention_mask3.repeat([batch_size,1,1,1])
        # ######5.build positionid
        position_ids = torch.arange(0, seq_length-1 , dtype=torch.long)
        position_ids2 = torch.arange(1, seq_length , dtype=torch.long)
        position_ids2 = torch.cat((position_ids,position_ids2),dim=-1).unsqueeze(0)     
        # #####6.build  the new input
        embed2 = torch.cat((outputs[0],embedtrigram[:,1:]),dim=-2)
        for i in self.fast_layer1:
            embed2 = i(embed2 ,attention_mask = attention_mask3,position_ids= position_ids2)
            embed2 =embed2[0]
        # #######7.intercept seq_len-1 ,due to the tiny defect of trigram,there is no trigram of  0，1 token in fact，so the 0,1 output is not valid
        output2 = embed2[:,0:seq_length-1]
        #import pdb;pdb.set_trace()
        loss_fct = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        #import pdb;pdb.set_trace();
        ######8.teacher and student trick
        hsloss =loss_fct( outputs[0][:,3:].clone(),output2[:,2:-1])  
        ####algorithm2 2.1.预测t2的准确率
        """实现上来说，i+1 layerN用fastlayerN代替，然后获取i+2 fastlayerN ,预测i+3的layerN
            在训练上，只能说再次拼接i+2 fastlayerN,长度为3seq_length-3,由于涉及到投机过程，取top5之类的，训练非常麻烦，所以暂时不实现
        """
        ####2.2 获得预测label，
        #predict_layerN2 = torch.cat((output2[:,:-1],output2[:,1:]),dim=-1)#######预测第二个token,长度为seq-2 
        # TODO: Consider parallelizing this loop for efficiency?
        chimera_logits = []
        for i in range(self.chimera_heads):           
            chimera_logits.append(self.chimera_head[i](output2 ))
        chimera_logits.append(orig[:,:])
        if output_orig:
            return torch.stack(chimera_logits, dim=0), outputs, orig
        #####对齐预测分布
        hsloss +=loss_fct( orig[:,3:].clone(),chimera_logits[0][:,2:-1]) 
        return {"logits":torch.stack(chimera_logits, dim=0),"hsloss":hsloss}
       
    def init_tree(self):
        self.tree = mc_sim_7b_63
        self.tree_buffer=generate_tree_buffers(self.tree,self.embed_tokens.weight.device)
        
    def repeat_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0].repeat(numr,1,1,1),i[1].repeat(numr,1,1,1)))
        return tuple(newkv)

    @torch.no_grad()
    def reduce_kv(self,kv,numr):
        newkv=[]
        for i in kv:
            newkv.append((i[0][:numr],i[1][:numr]))
        return tuple(newkv)


    def reset_kv(self):
        self.fast_layer0_kv=None 
        self.fast_layer1_kv=None
        
    def chimera_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Chimera
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        chimera_choices=[1, 7, 6],
        posterior_threshold=0.09,  # threshold validation of Chimera output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            chimera_choices (list, optional): A list of integers indicating the number of choices for each Chimera head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache chimera buffers (the fixed patterns for tree attention)
        if hasattr(self, "chimera_choices") and self.chimera_choices == chimera_choices:
            # Load the cached chimera buffer
            chimera_buffers = self.chimera_buffers
        else:
            # Initialize the chimera buffer
            chimera_buffers = generate_chimera_buffers(
                chimera_choices, device=self.base_model.device
            )
        self.chimera_buffers = chimera_buffers
        self.chimera_choices = chimera_choices

        chimera_topk = chimera_choices[1:]

        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self.base_model)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]

        reset_chimera_mode(self)
        # Initialize tree attention mask and process prefill tokens
        chimera_logits, logits = initialize_chimera(
            input_ids, self, chimera_buffers["chimera_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Chimera heads
            candidates, tree_candidates = generate_candidates(
                chimera_logits,
                logits,
                chimera_topk,
                chimera_buffers["tree_indices"],
                temperature,
            )

            # Use tree attention to verify the candidates and get predictions
            chimera_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                chimera_buffers["chimera_position_ids"],
                input_ids,
                chimera_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )

            # Update the input_ids and logits
            input_ids, logits, chimera_logits, new_token = update_inference_inputs(
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
            )

            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                break
