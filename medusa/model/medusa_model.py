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

class SingleMedusa():
    def __init__(self,model,medusa_head_name_or_path ='../../model/medusa_8_1_token' , medusa_num_heads = 8,medusa_num_layers=1):
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        
        self.medusa_num_heads = medusa_num_heads
        ####加载medusa头
        self.medusa_head = nn.ModuleList(
                    [
                        nn.Sequential(
                            *([ResBlock(self.hidden_size)] * medusa_num_layers),
                            nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                        )
                        for _ in range(medusa_num_heads)
                    ]
                )
        # Ensure medusa_head's dtype and device align with the base_model
        self.medusa_head.to(model.base_model.dtype).to(model.base_model.device)
        # for i in range(medusa_num_heads):
        # # Initialize the weights of each medusa_head using the base model's weights
        #     self.medusa_head[i][-1].weight.data[:] = model.base_model.lm_head.weight.data[:]
        

        #x = torch.randn(32, 2, 4096)
        medusa_head_path = os.path.join(medusa_head_name_or_path, "medusa_lm_head.pt")
        if os.path.exists(medusa_head_path):
            filename = medusa_head_path
        else:
            filename = hf_hub_download(medusa_head_name_or_path, "medusa_lm_head.pt")
        medusa_head_state_dict = torch.load(filename, map_location=model.base_model.device)
        self.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)

       
class MedusaConfig(PretrainedConfig):
    def __init__(
        self,
        medusa_num_heads=2,
        medusa_num_layers=1,
        base_model_name_or_path='../model/vicuna-7b-v1.3',#"lmsys/vicuna-7b-v1.3",'../model/vicuna-7b-v1.3', #'../../../../model/vicuna-7b-v1.3',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.medusa_num_heads = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
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


class MedusaModel(nn.Module):
    """The Medusa Language Model Head.

    This module creates a series of prediction heads (based on the 'medusa' parameter)
    on top of a given base model. Each head is composed of a sequence of residual blocks
    followed by a linear layer.
    """

    def __init__(
        self,
        base_model,
        medusa_num_heads=2,
        medusa_num_layers=1,
        base_model_name_or_path='../../../model/vicuna-7b-v1.3',#"lmsys/vicuna-7b-v1.3",'../model/vicuna-7b-v1.3',#'../../../../model/vicuna-7b-v1.3'
    ):
        """
        Args:
            base_model (nn.Module): The base language model to be used.
            medusa_num_heads (int, optional): Number of additional tokens to predict. Defaults to 3.
            medusa_num_layers (int, optional): Number of ResBlock layers for each Medusa head. Defaults to 0.
        """
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.hidden_size = base_model.lm_head.weight.shape[-1]
        self.vocab_size = base_model.lm_head.weight.shape[0]
        self.medusa = medusa_num_heads
        self.medusa_num_layers = medusa_num_layers
        self.base_model_name_or_path =base_model_name_or_path#'../../../../../model/vicuna-7b-v1.3'#base_model_name_or_path #'../../../../model/vicuna-7b-v1.3' #
        print('path: ',self.base_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name_or_path)
        # Create a list of Medusa heads
        
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(self.hidden_size)] * medusa_num_layers),
                    nn.Linear(self.hidden_size, self.vocab_size, bias=False),
                )
                for _ in range(medusa_num_heads)
            ]
        )
        
        import copy
        #self.grulayer = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1,batch_first=True)
        self.trimlp = nn.Sequential(
                    *([ResBlock(self.hidden_size*3)] ),
                    nn.Linear(self.hidden_size*3, self.hidden_size, bias=False),
                )#copy.deepcopy(base_model.model.layers[0])
        # self.fast_layer1 = nn.Sequential(copy.deepcopy(base_model.model.layers[0]),
        #                                  copy.deepcopy(base_model.model.layers[4]),
        #                                  copy.deepcopy(base_model.model.layers[-4]),
        #                                   copy.deepcopy(base_model.model.layers[-1])
        #                                 )
        newconfig = base_model.config
        newconfig.hidden_size = newconfig.hidden_size*2
        self.fast_layer1 = nn.Sequential(
                                         copy.deepcopy(base_model.model.layers[4]),
                                          copy.deepcopy(base_model.model.layers[8]),
                                         copy.deepcopy(base_model.model.layers[-4]),
                                          copy.deepcopy(base_model.model.layers[-1])
                                        )
        self.fastoutput = nn.Sequential(
                    *([ResBlock(self.hidden_size*2)] ),
                    nn.Linear(self.hidden_size*2, self.hidden_size, bias=False),
                )

       # nn.Sequential()
        for param in self.fast_layer1.parameters():
            param.require_grad = True

        self.medusa_head.to(self.base_model.dtype).to(self.base_model.device)
        self.fast_layer1.to(self.base_model.dtype).to(self.base_model.device)
        self.trimlp.to(self.base_model.dtype).to(self.base_model.device)
        self.fastoutput.to(self.base_model.dtype).to(self.base_model.device)

        # for param in self.fast_layer.parameters():
        #     param.requires_grad = True
        for i in range(medusa_num_heads):
            # Initialize the weights of each medusa_head using the base model's weights
            #self.medusa_head[i][-1].weight.data[:] = base_model.lm_head.weight.data[:]
            self.medusa_head[i][-1].weight.data[:]=base_model.lm_head.weight.data
            #torch.cat((base_model.lm_head.weight.data,base_model.lm_head.weight.data),dim=-1)
            #torch.cat((base_model.lm_head.weight.data),dim=-1)#torch.cat((base_model.lm_head.weight.data,base_model.lm_head.weight.data),dim=-1),base_model.lm_head.weight.data ,base_model.lm_head.weight.data
            #,base_model.lm_head.weight.data
    def get_tokenizer(self):
        """Get the tokenizer of the base model.

        Returns:
            Tokenizer: The tokenizer of the base model.
        """
        return self.tokenizer

    @classmethod
    def from_pretrained(
        cls,
        medusa_head_name_or_path,
        **kwargs,
    ):
        """
        Args:
            medusa_head_name_or_path (str): Name or path of the Medusa head to load.
            **kwargs: Additional keyword arguments for loading the base model.

        Returns:
            MedusaModel: A MedusaModel instance loaded from the given path.
        """
        medusa_config = MedusaConfig.from_pretrained(medusa_head_name_or_path)
        base_model = KVLlamaForCausalLM.from_pretrained(
            medusa_config.base_model_name_or_path, **kwargs
        )
        print("path",medusa_config.base_model_name_or_path)
        model = cls(
            base_model,
            medusa_config.medusa_num_heads,
            medusa_config.medusa_num_layers,
            medusa_config.base_model_name_or_path,
            
        )
        medusa_head_path = os.path.join(medusa_head_name_or_path, "medusa_lm_head.pt")
        if os.path.exists(medusa_head_path):
            filename = medusa_head_path
        else:
            filename = hf_hub_download(medusa_head_name_or_path, "medusa_lm_head.pt")
        medusa_head_state_dict = torch.load(filename, map_location=base_model.device)
        model.medusa_head.load_state_dict(medusa_head_state_dict, strict=False)
        #trimlp_head_path = os.path.join(medusa_head_name_or_path, "medusa_lm_head.pt")
        #model.trimlp.load_state_dict()
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        
        with torch.inference_mode():
            # Pass input through the base model
            outputs = self.base_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                #output_hidden_states=True,
            )
            if output_orig:
                orig = self.base_model.lm_head(outputs[0])
            orig = self.base_model.lm_head(outputs[0])
        #########1/for 循环，取i=1,2,3....n#######
        # new_hs = outputs['hidden_states'][1][0][0].unsqueeze(0)
        # for i  in range(1,len(outputs['hidden_states'][1][0])):
        #     #hs1 = outputs['hidden_states'][-2][0][i]
        #     #hs0 = torch.cat(( outputs['hidden_states'][-2][0][:i-1],hs1.unsqueeze(0)), dim=0)
        #     hs0 = torch.cat(( outputs['hidden_states'][-2][0][:i],outputs['hidden_states'][1][0][i].unsqueeze(0)), dim=0)
        #     #hs0 = outputs['hidden_states'][-2][0][:i]
        #     hs_final = self.fast_layer(hs0.unsqueeze(0))
        #     new_hs=torch.cat((new_hs,hs_final[0][0][-2].unsqueeze(0)),dim=0)


        ##########采用linear做注意力融合直接输出#####
        ########获得拼接后的hidden_states
        # embed =self.base_model.model.embed_tokens(input_ids)
        # from modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
        # batch_size, seq_length = embed.shape[:2]
        # attention_mask = _prepare_4d_causal_attention_mask(
        #                  attention_mask, (batch_size, seq_length), embed, 0
        #             )
        # output2 = self.base_model.model.layers[0](embed,attention_mask = attention_mask)
        # new_hs=torch.cat((outputs['hidden_states'][-1][0,:-1],output2[0][0,1:],outputs['hidden_states'][1][0,2:]),dim=-1).unsqueeze(0)
        # for i in range(1,len(outputs['hidden_states'][-1])):
        #     h1 = torch.cat((outputs['hidden_states'][-1][i,:-1],output2[0][i,1:],outputs['hidden_states'][1][i,2:]),dim=-1).unsqueeze(0)
        #     new_hs = torch.cat((new_hs,h1),dim=0)
            
        # ###############将hidden_states 2046,12288 送入模型
        # new_hs=torch.cat((outputs['hidden_states'][-1][0,:-1],outputs['hidden_states'][16][0,1:]),dim=-1).unsqueeze(0)
        # for i in range(1,len(outputs['hidden_states'][-1])):
        #     h1 = torch.cat((outputs['hidden_states'][-1][i,:-1],outputs['hidden_states'][16][i,1:]),dim=-1).unsqueeze(0)
        #     new_hs = torch.cat((new_hs,h1),dim=0)


        ####获得trigram layerN输出
        
        # trihs = torch.cat((input_ids[:,:-2].unsqueeze(0),input_ids[:,1:-1].unsqueeze(0),input_ids[:,2:].unsqueeze(0)),dim=0)
        # trihs = torch.transpose(trihs,dim0=0,dim1=2)
        # trihs = torch.transpose(trihs,dim0=0,dim1=1)
        # batch_size =  trihs.shape[0]
        # hs_size = self.hidden_size
        # trihs = torch.flatten(trihs,end_dim=1)
        # with torch.inference_mode():
        #     trihs =self.base_model.model.embed_tokens(trihs)
            
        #     #from modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
        #     #trihs =  self.base_model.model(input_ids= trihs)
        #     trihs = self.base_model.model.layers[0](trihs)
        #     trihs = self.base_model.model.layers[1](trihs[0])
        #     trihs = self.base_model.model.layers[2](trihs[0])
            
        # trihs= trihs[0]
        
        # trihs = trihs.view((batch_size,-1,3,self.hidden_size))
        # trihs =trihs[:,:,-1,:] #得到每一个token混合后的输出
        # trihs = torch.cat((trihs[:,:1],trihs),dim=-2)

        #####1.get trigram#####
        embed =self.base_model.model.embed_tokens(input_ids)
        embedtrigram = torch.cat((embed[:,:-2],embed[:,1:-1],embed[:,2:]),dim=-1)
        #gram0 = torch.cat((embed[:,0],embed[:,0],embed[:,0]),dim=-1).unsqueeze(1)
        gram1 = torch.cat((embed[:,0],embed[:,1],embed[:,1]),dim=-1).unsqueeze(1)
        embedtrigram = torch.cat((gram1,embedtrigram),dim=-2)
        embed = self.trimlp(embedtrigram )
        
        from modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
        batch_size, seq_length = embed.shape[:2]
        attention_mask = _prepare_4d_causal_attention_mask(
                         attention_mask[:,:-1], (batch_size, seq_length), embed, 0
                    )
        attention_mask  = attention_mask.to(self.base_model.device)
        # embedtrigram = torch.cat((embed[:,:-2],embed[:,1:-1],embed[:,2:]),dim=-1)
        #for  i in self.fast_layer :   
        #input1 = torch.cat((outputs[0][:,:-1],embed[:,:]),dim=-1)
        for i in self.fast_layer1:
            embed =    i(embed,attention_mask = attention_mask)
            embed =    embed[0]
        predict_layerN = torch.cat((embed,outputs[0][:,:-1]),dim=-1)
        predict_layerN  = self.fastoutput(predict_layerN)
        #import pdb;pdb.set_trace();
        # output3 = self.fast_layer3(output3[0]  ,attention_mask = attention_mask)
        # output3 = self.fast_layer4(output3[0]  ,attention_mask = attention_mask)
        # output3 = self.fast_layer5(output3[0]  ,attention_mask = attention_mask)
        
        #output3 = self.W(output3[0])
        #output3 = self.W(output3[0])
        #output3 = self.base_model.model.norm(output3[0])
        # embed = self.base_model.model.layers[0](embed,attention_mask = attention_mask)
        # #embed = self.fast_layer1(embed[0],attention_mask = attention_mask)#####new token的fastlayer1输出
        
        # embed = embed[0]
        # embed = self.fitlayer(embed)
        # ########
        #output2 = self.fast_layer2(output2[0],attention_mask = attention_mask)
        # #####2.构造新的attention_mask,2seq_length-1
        # attention_mask2 =torch.full((seq_length, seq_length), -3.4028e+38) + torch.diag(torch.zeros(seq_length)+3.4028e+38-1)
        # attention_mask2 = attention_mask2.to(self.base_model.device)
        # attention_mask3 = torch.cat((attention_mask[0,0,:,:],attention_mask2),dim=-1)
        # attention_mask3 = torch.cat((attention_mask3[:,:-1],attention_mask3[:-1,:-1]),dim=-2).unsqueeze(0).unsqueeze(0)
        # attention_mask3 = attention_mask3.repeat([batch_size,1,1,1])
        # ######3.构造positionid
        # position_ids = torch.arange(0, seq_length , dtype=torch.long)
        # position_ids2 = torch.arange(1, seq_length , dtype=torch.long)
        # position_ids2 = torch.cat((position_ids,position_ids2),dim=-1).unsqueeze(0)

        
        # #####4.构造新的input,计算结果
        # embed2 = torch.cat((output3[0],embed[:,1:]),dim=-2)
        # #embed2 = torch.cat((outputs[0],trihs),dim=-2)
        # # # ######首先
        # output2 = self.fast_layer2(embed2 ,attention_mask = attention_mask3,position_ids= position_ids2)
        # output2 = self.fast_layer3(output2[0] ,attention_mask = attention_mask3,position_ids= position_ids2)
        # #######4.2和最后一层拼接进行计算
        
        # output3 = self.fast_layer1(output[0] ,attention_mask = attention_mask3,position_ids= position_ids2)
        #########5得到最终注意力融合结果
        #hs_norm = self.base_model.model.norm(output3[0][:,-seq_length+1:])
        ########将结果和i拼接
        #new_hs=torch.cat((outputs[0][:,:-1],output3[0][:,-seq_length+1:]),dim=-1)
        #new_hs = output3[0][:,-seq_length+1:]
        #output2 = self.fast_layer2(output2[0],attention_mask = attention_mask)
        ######获取grulayer
        #res = torch.ones((outputs[0].shape[0],outputs[0].shape[1]-2,3,outputs[0].shape[2]))
        
        # res[:,:,0,:] = outputs[0][:,:-2]
        # res[:,:,1,:] = outputs[0][:,1:-1]
        # res[:,:,2,:] = output2[0][:,2:]
        # res = torch.flatten(res,start_dim=-2)
        ###展平seqlen和原batchsize,作为输入
        #res=torch.flatten(res,end_dim=1)
        ######输入gru
        #res =res.to("cuda:0").contiguous()
        #torch.backends.cudnn.enabled = False
        
        # gruout,h_0 = self.grulayer(res)
        # #####先展平gru后两个，讲2048->4096，现在大小为（batch*seq,4096）
        # gruout=torch.flatten(gruout,start_dim=-2)
        # #######现在讲batchsize恢复
        # gruout = gruout.view(batch_size,seq_length-2,-1)####(batch,seq-1,hidden)
        # new_hs=torch.cat((outputs[0][0,:-1],output2[0][0,1:]),dim=-1).unsqueeze(0)
        # for i in range(1,len(outputs[0])):
        #     h1 =torch.cat((outputs[0][i,:-1],output2[0][i,1:]),dim=-1).unsqueeze(0)
        #     new_hs = torch.cat((new_hs,h1),dim=0)
        
        
            # Pass input through the base model
            # trioutputs = self.base_model.model(
            #     input_ids=trihs
            #     # attention_mask=attention_mask,
            #     # past_key_values=past_key_values,
            #     # position_ids=position_ids,
            #     #output_hidden_states=True,
            # )
        ################teacher and student###################
        loss_fct = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        #import pdb;
        #pdb.set_trace();
        hsloss =loss_fct( outputs[0][:,2:].clone(),predict_layerN[:,1:])  
        ###############################################
        
        #embed3 = torch.cat((outputs[0][:,1:-2],embed[:,:-1]),dim=-1)#output2[0][:,-seq_length+2:-1]
        ####预测t2的准确率
        ####假设token是正确的#####然后把layerN的计算替换为fastlayerN
        
        predict_layerN2 = torch.cat((embed[:,1:],embed[:,:-1]),dim=-1)
        predict_layerN2 = self.fastoutput(predict_layerN2)

        
        # TODO: Consider parallelizing this loop for efficiency?
        medusa_logits = []
        for i in range(self.medusa):
            #######修改后输出######
            medusa_logits.append(self.medusa_head[i](predict_layerN )[:,1:])#self.medusa_head[i]embed3.unsqueeze(0)(outputs[0]))#hidden_states[i*4].clone()))gruout.to(self.base_model.dtype)
        medusa_logits.append(self.medusa_head[0](predict_layerN2 ))
            ######原输出######
            #medusa_logits.append(self.medusa_head[i]((outputs[0].clone())))
        # if output_orig:
        #     return torch.stack(medusa_logits, dim=0), outputs, orig
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        hsloss +=loss_fct( orig[:,2:].clone(),medusa_logits[0][:,:]) 
        return {"logits":torch.stack(medusa_logits, dim=0),"hsloss":hsloss}
       

    def medusa_generate(
        self,
        input_ids,
        attention_mask=None,
        temperature=0.0,
        max_steps=512,
        # The hyperparameters below are for the Medusa
        # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
        medusa_choices=[1, 7, 6],
        posterior_threshold=0.09,  # threshold validation of Medusa output
        # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
        posterior_alpha=0.3,
    ):
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()

        # Cache medusa buffers (the fixed patterns for tree attention)
        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices

        medusa_topk = medusa_choices[1:]

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

        reset_medusa_mode(self)
        # Initialize tree attention mask and process prefill tokens
        medusa_logits, logits = initialize_medusa(
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )

        new_token = 0
        last_round_token = 0

        for idx in range(max_steps):
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_topk,
                medusa_buffers["tree_indices"],
                temperature,
            )

            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )

            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha
            )

            # Update the input_ids and logits
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                outputs,
                logits,
                medusa_logits,
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
