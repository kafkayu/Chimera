import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
#from .modeling_llama_kv import LlamaForCausalLM as KVLlamaForCausalLM
from transformers import AutoTokenizer ,AutoModelForCausalLM

from .utils import *
from .kv_cache import initialize_past_key_values
import os
from huggingface_hub import hf_hub_download
import copy
from torch.nn import CrossEntropyLoss,MSELoss
from transformers.models.llama.modeling_llama import  LlamaModel,LlamaDecoderLayer
from .choices import *
from .modeling_attn_mask_utils import AttentionMaskConverter, _prepare_4d_causal_attention_mask
import wandb
from .cnet import Model
wandb.login(key="6224ac7517be176065dbe00432983a2ef90fa010")#######wandb key
# Import the summary writer 
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object 
writer = SummaryWriter()
       
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
        self.chimera_head = nn.ModuleList(
            [
                nn.Sequential(
                    # *([ResBlock(self.hidden_size)] * chimera_num_layers),
                    nn.Linear(self.hidden_size*2, self.vocab_size, bias=False),
                )
                for _ in range(self.chimera_heads)
            ]
        )
        
        import copy

        self.trimlp = nn.Sequential(
                    *([ResBlock(self.hidden_size*3)] ),
                    nn.Linear(self.hidden_size*3, self.hidden_size, bias=False),
                )
        config = copy.deepcopy(self.base_model.config)
        config.num_hidden_layers = 1

        self.fast_layer1 = nn.Sequential(
                                        LlamaDecoderLayer(config)
                                        # copy.deepcopy(base_model.model.layers[-1])
                                        )


        for param in self.fast_layer1.parameters():
            param.require_grad = True
        self.chimera_head.to(self.base_model.dtype).to(self.base_model.device)
        self.fast_layer1.to(self.base_model.dtype).to(self.base_model.device)
        self.trimlp.to(self.base_model.dtype).to(self.base_model.device)
        for param in self.base_model.parameters():
            param.require_grad = False
        for param in self.base_model.lm_head.parameters():
            param.require_grad = False


        for i in range(chimera_num_heads):
            # Initialize the weights of each chimera_head using the base model's weights
            self.chimera_head[i][-1].weight.data[:]=torch.cat((base_model.lm_head.weight.data,base_model.lm_head.weight.data),dim=-1)
            
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
        base_model = AutoModelForCausalLM.from_pretrained(
            chimera_config.base_model_name_or_path
        )
        #KVLlamaForCausalLM.from_pretrained(
        #     chimera_config.base_model_name_or_path, **kwargs
        # )
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
        
        #2.fast_layer0
        # chimera_fast_layer0_path = os.path.join(chimera_name_or_path, "fast_layer0.pt")
        # if os.path.exists(chimera_fast_layer0_path):
        #     filename = chimera_fast_layer0_path
        # else:
        #     filename = hf_hub_download(chimera_name_or_path, "fast_layer0.pt")
        # chimera_state_dict = torch.load(filename, map_location=base_model.device)
        # model.fast_layer0.load_state_dict(chimera_state_dict, strict=False)
        
        ##3.fast_layer1
        chimera_fast_layer1_path = os.path.join(chimera_name_or_path, "fast_layer1.pt")
        if os.path.exists(chimera_fast_layer1_path):
            filename = chimera_fast_layer1_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "fast_layer1.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        model.fast_layer1.load_state_dict(chimera_state_dict, strict=False)

        ##3.fast_layer2
        # chimera_fast_layer2_path = os.path.join(chimera_name_or_path, "fast_layer2.pt")
        # if os.path.exists(chimera_fast_layer2_path):
        #     filename = chimera_fast_layer2_path
        # else:
        #     filename = hf_hub_download(chimera_name_or_path, "fast_layer2.pt")
        # chimera_state_dict = torch.load(filename, map_location=base_model.device)
        # model.fast_layer2.load_state_dict(chimera_state_dict, strict=False)
        
        #4.chimera_head
        chimera_head_path = os.path.join(chimera_name_or_path, "chimera_head.pt")
        if os.path.exists(chimera_head_path):
            filename = chimera_head_path
        else:
            filename = hf_hub_download(chimera_name_or_path, "chimera_head.pt")
        chimera_state_dict = torch.load(filename, map_location=base_model.device)
        model.chimera_head.load_state_dict(chimera_state_dict, strict=False)
        
        return model


        
    def load_chimera(
        self,
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

        #KVLlamaForCausalLM.from_pretrained(
        #     chimera_config.base_model_name_or_path, **kwargs
        # )
        # print("path",chimera_config.base_model_name_or_path)
        ##1.trimlp layer
        chimera_trimlp_path = os.path.join(chimera_name_or_path, "trimlp.pt")
        chimera_state_dict = torch.load(chimera_trimlp_path)
        self.trimlp.load_state_dict(chimera_state_dict)
        
        
        #3.fast_layer1
        chimera_fast_layer1_path = os.path.join(chimera_name_or_path, "fast_layer1.pt")
        chimera_state_dict = torch.load(chimera_fast_layer1_path)
        self.fast_layer1.load_state_dict(chimera_state_dict)
        
        # ##4.chimera_head
        # chimera_head_path = os.path.join(chimera_name_or_path, "chimera_head.pt")
        # chimera_state_dict = torch.load(chimera_head_path)
        # self.chimera_head.load_state_dict(chimera_state_dict)
        
        return self
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
        embed1 = self.trimlp(embedtrigram )
        
        batch_size, seq_length = embed1.shape[:2]
        attention_mask = _prepare_4d_causal_attention_mask(
                         attention_mask[:,:], (1, seq_length-1), embed[:,:-1], 0
                    )
        attention_mask  = attention_mask.to(self.base_model.device)

        
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
        mask_value=attention_mask[0,0,0,1].cpu()
        attention_mask2 =torch.full((seq_length-1, seq_length-1), mask_value) + torch.diag(torch.zeros(seq_length-1)+mask_value-1)
        attention_mask2 = attention_mask2.to(self.base_model.device)

        
        attention_mask3 = torch.cat((attention_mask[0,0],attention_mask[0,0]),dim=-1)
        attention_mask3 = torch.cat((attention_mask3,attention_mask3),dim=-2).unsqueeze(0).unsqueeze(0)
        attention_mask3 = attention_mask3.repeat([batch_size,1,1,1])
        
        # ######5.build positionid
        # import pdb;pdb.set_trace();
        position_ids = torch.arange(0, seq_length-1 , dtype=torch.long)
        position_ids2 = torch.arange(1, seq_length , dtype=torch.long)
        position_ids2 = torch.cat((position_ids,position_ids2),dim=-1).unsqueeze(0)  
        # import pdb;pdb.set_trace()
        # #####6.build  the new input  
        embed2 = torch.cat((outputs[0],embed1[:,1:]),dim=-2)
        for i in self.fast_layer1:
            embed2 = i(embed2 ,attention_mask = attention_mask3,position_ids= position_ids2)
            embed2 =embed2[0]
        #######7.intercept seq_len-1 ,due to the tiny defect of trigram,there is no trigram of  0，1 token in fact，so the 0,1 output is not valid
        output2 = embed2[:,-seq_length+1:]
        # import pdb;pdb.set_trace();
        loss_fct =torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean') 
        ######8.teacher and student trick
        hsloss =loss_fct( outputs[0][:,1:].clone(),output2[:,:-1])  
        ####algorithm2 2.1.预测t2的准确率
        """实现上来说，i+1 layerN用fastlayerN代替，然后获取i+2 fastlayerN ,预测i+3的layerN
            在训练上，只能说再次拼接i+2 fastlayerN,长度为3seq_length-3,由于涉及到投机过程，取top5之类的，训练非常麻烦，所以暂时不实现
        """
        # ##### 8.预测t3 seq-1 + seq-1 + seq-1 ，因为embed不够了
        
        embed1 = torch.cat((embed1,embed1[:,-1].unsqueeze(1)),dim=-2)
        attention_mask3 = torch.cat((attention_mask[0,0],attention_mask2,attention_mask[0,0]),dim=-1)
        attention_mask3 = torch.cat((attention_mask3,attention_mask3,attention_mask3),dim=-2).unsqueeze(0).unsqueeze(0)
        attention_mask3 = attention_mask3.repeat([batch_size,1,1,1])        
        position_ids = torch.arange(0, seq_length-1 , dtype=torch.long)
        position_ids2 = torch.arange(1, seq_length , dtype=torch.long)
        position_ids3 = torch.arange(2, seq_length+1 , dtype=torch.long)
        position_ids3 = torch.cat((position_ids,position_ids2,position_ids3),dim=-1).unsqueeze(0)   
        embed3 = torch.cat((outputs[0][:,:],output2,embed1[:,2:]),dim=-2).clone().detach() 
        for i in self.fast_layer1:
            embed3 = i(embed3 ,attention_mask = attention_mask3,position_ids= position_ids3)
            embed3 =embed3[0]
        # #######7.intercept seq_len-1 ,due to the tiny defect of trigram,there is no trigram of  0，1 token in fact，so the 0,1 output is not valid
        output3 = embed3[:,-seq_length+1:]
        # import pdb;pdb.set_trace()
        ###i+4
        embed1 = torch.cat((embed1,embed1[:,-1].unsqueeze(1)),dim=-2)
        attention_mask3 = torch.cat((attention_mask[0,0],attention_mask2,attention_mask2,attention_mask[0,0]),dim=-1)
        attention_mask3 = torch.cat((attention_mask3,attention_mask3,attention_mask3,attention_mask3),dim=-2).unsqueeze(0).unsqueeze(0)
        attention_mask3 = attention_mask3.repeat([batch_size,1,1,1])        
        position_ids4 = torch.arange(3, seq_length+2 , dtype=torch.long)
        position_ids4 = torch.cat((position_ids3[0],position_ids4),dim=-1).unsqueeze(0)   
        embed3 = torch.cat((outputs[0][:,:],output2,output3,embed1[:,3:]),dim=-2).clone().detach() 
        for i in self.fast_layer1:
            embed3 = i(embed3 ,attention_mask = attention_mask3,position_ids= position_ids4)
            embed3 =embed3[0]
        # #######7.intercept seq_len-1 ,due to the tiny defect of trigram,there is no trigram of  0，1 token in fact，so the 0,1 output is not valid
        output4 = embed3[:,-seq_length+1:]
        ### i+5
        embed1 = torch.cat((embed1,embed1[:,-1].unsqueeze(1)),dim=-2)
        attention_mask3 = torch.cat((attention_mask[0,0],attention_mask2,attention_mask2,attention_mask2,attention_mask[0,0]),dim=-1)
        attention_mask3 = torch.cat((attention_mask3,attention_mask3,attention_mask3,attention_mask3,attention_mask3),dim=-2).unsqueeze(0).unsqueeze(0)
        attention_mask3 = attention_mask3.repeat([batch_size,1,1,1])        
        position_ids5 = torch.arange(4, seq_length+3 , dtype=torch.long)
        position_ids5 = torch.cat((position_ids4[0],position_ids5),dim=-1).unsqueeze(0)   
        embed3 = torch.cat((outputs[0][:,:],output2,output3,output4,embed1[:,4:]),dim=-2).clone().detach() 
        for i in self.fast_layer1:
            embed3 = i(embed3 ,attention_mask = attention_mask3,position_ids= position_ids5)
            embed3 =embed3[0]
        # #######7.intercept seq_len-1 ,due to the tiny defect of trigram,there is no trigram of  0，1 token in fact，so the 0,1 output is not valid
        output5 = embed3[:,-seq_length+1:]
        
        
        
        ####2.2 获得预测label，
        #predict_layerN2 = torch.cat((output2[:,:-1],output2[:,1:]),dim=-1)#######预测第二个token,长度为seq-2 
        output22 = torch.cat((outputs[0],output2),dim=-1).detach().requires_grad_()
        output33 = torch.cat((outputs[0],output3),dim=-1).detach().requires_grad_()
        output44 = torch.cat((outputs[0],output4),dim=-1).detach().requires_grad_()
        output55 = torch.cat((outputs[0],output5),dim=-1).detach().requires_grad_()
        
        chimera_logits = []
        for i in range(self.chimera_heads): 
           if i== 0 :chimera_logits.append(self.chimera_head[i]((output22)))
           elif i==1 :chimera_logits.append(self.chimera_head[i]((output33)))
           elif i==2 :chimera_logits.append(self.chimera_head[i]((output44)))
           elif i==3 :chimera_logits.append(self.chimera_head[i]((output55)))
               
           hsloss +=loss_fct( orig[:,3+i:].clone(),chimera_logits[i][:,2:-1-i])
        chimera_logits.append(orig[:,:])
        if output_orig:
            return torch.stack(chimera_logits, dim=0), outputs, orig
        #####对齐预测分布
        
        

        
        return {"logits":torch.stack(chimera_logits, dim=0),"hsloss":hsloss}
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
        use_cache = False,
        cachedata = {} ,
        
    ):
        cachedata ={}
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
        #####1.get trigram#####
        if use_cache ==True:
            
            newt1 = self.trimlp(self.base_model.model.embed_tokens(input_ids[:,-3:]))
            embedtrigram = newt1
        else:
            embed =self.base_model.model.embed_tokens(input_ids)
            embedtrigram = torch.cat((embed[:,:-2],embed[:,1:-1],embed[:,2:]),dim=-1)
            gram0 = torch.cat((embed[:,0],embed[:,0],embed[:,0]),dim=-1).unsqueeze(1)
            gram1 = torch.cat((embed[:,0],embed[:,1],embed[:,1]),dim=-1).unsqueeze(1)
            embedtrigram = torch.cat((gram0,gram1,embedtrigram),dim=-2)
            embedtrigram = self.trimlp(embedtrigram )
        
        batch_size, seq_length = embedtrigram.shape[:2]
        attention_mask = _prepare_4d_causal_attention_mask(
                         attention_mask[:,:], (batch_size, seq_length), embed, 0
                    )
        attention_mask  = attention_mask.to(self.base_model.device)
        ########1.2 forward融合信息
        if use_cache == True :
            for i in self.fast_layer0:
                embedtrigram = i(embedtrigram,attention_mask =attention_mask ,use_cache=use_cache,past_key_value =cache_data['fast_layer0'])
                embedtrigram = embedtrigram[0]
                cache_data['fast_layer0'] = embedtrigram['past_key_value']
        else:
            for i in self.fast_layer0:
                embedtrigram = i(embedtrigram,attention_mask =attention_mask ,use_cache=use_cache)
        
        # #####2.构造新的attention_mask,seq_length
        # # ######3.构造positionid
        position_ids = torch.arange(0, seq_length, dtype=torch.long).unsqueeze(0)
        # #####4.构造新的input,计算结果
        embed2 = torch.cat((last_hs,embedtrigram[:,-1].unsqueeze(1)),dim=-2)
        attention_mask[:,:,-2,-1] = 0 
        if use_cache == True:
            for i in self.fast_layer1:
                embed2 = i(embed2,attention_mask =attention_mask ,use_cache=use_cache,past_key_value =cache_data['fast_layer1'])
                cache_data['fast_layer1'] = embed2['past_key_value']
                embed2 = embed2[0]
        else:
            for i in self.fast_layer1:
                embed2 = i(embed2 ,attention_mask = attention_mask )
                embed2 =embed2[0]
        # #######4.2和最后一层拼接进行计算
        output2 = embed2
        #########5.将layerN拼接作为输入预测据说效果更好 大小为seq-1，由于0，1没有trigram，实际上只有2开始有效
        chimera_logits=self.chimera_head[0](embed2[:,-2] )
        if use_cache:
             return chimera_logits,embed2[:,-2]  ,cache_data
        return chimera_logits[0],embed2[:,-2]  
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
        for i in range(max_len):
            ####3. forward
            orig ,fs_hs  = self.fastlayer_forward(input_ids=input_ids,attention_mask=attention_mask,last_hs = last_hs)
            #####4.拼接新last_hs
            last_hs = torch.cat((last_hs,fs_hs.unsqueeze(1)),dim=-2)
            last_hs = last_hs.repeat_interleave(choices,dim=0)
            ####5.获取新t0
            pro,t0 = orig.topk(k=choices,dim=-1)
            t0 = t0.unsqueeze(-1)        
            ######6.拼接,获得新的batch
            input_ids =  input_ids.unsqueeze(1)
            input_ids =  input_ids.repeat_interleave(choices,dim=1)
            input_ids =  torch.cat((input_ids,t0),dim=-1)
            #print(input_ids.shape)
            input_ids = input_ids.flatten(0,1)
            ######7.更新att_m
            #import pdb;pdb.set_trace();
            attention_mask = torch.cat((attention_mask,attention_mask[:,0].unsqueeze(1)),dim=-1)
            attention_mask = attention_mask.repeat(choices,1)
        return input_ids 
    def headgenerate2(self,
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
        prob = torch.tensor([[1]])
        #candidate = input_ids[:,-1].unsqueeze(0)
        op = []
        for i in range(max_len):
            ####3. forward
            orig ,fs_hs  = self.fastlayer_forward(input_ids=input_ids,attention_mask=attention_mask,last_hs = last_hs)
            #####4.拼接新last_hs
            last_hs = torch.cat((last_hs,fs_hs.unsqueeze(1)),dim=-2)
            last_hs = last_hs.repeat_interleave(choices,dim=0)
            ####5.获取新t0
            probability = torch.softmax(orig,dim=-1)
            pro,t0 = probability.topk(k=choices,dim=-1)
            #import pdb;pdb.set_trace();
            t0 = t0.unsqueeze(-1)    
            pro = pro.unsqueeze(-1)
            ######6.拼接,获得新的batch
            input_ids =  input_ids.unsqueeze(1)
            prob = prob.unsqueeze(1)
            input_ids =  input_ids.repeat_interleave(choices,dim=1)
            prob = prob.repeat_interleave(choices,dim=1)
            input_ids =  torch.cat((input_ids,t0),dim=-1)
            prob =  torch.cat((prob,pro),dim=-1)
            #print(input_ids.shape)
            input_ids = input_ids.flatten(0,1)
            prob = prob.flatten(0,1)
            ######7.更新att_m
            #import pdb;pdb.set_trace();
            attention_mask = torch.cat((attention_mask,attention_mask[:,0].unsqueeze(1)),dim=-1)
            attention_mask = attention_mask.repeat(choices,1)
            op.append(probability)
        return input_ids ,prob,op
    def naive_predict(model,input_ids = None ,candidate = None ,attention_mask=None,past_key_values= None,topk=1):
         """candidate 是所有可能token序列片段
         """
         aclength = 0
         totallen = len(candidate[0])
         best_candidate = torch.tensor([])
         
         input_ids = input_ids.repeat(len(candidate),1)
         #for i in candidate:
         batch_index = 0
         attention_mask =    attention_mask.repeat(len(candidate),1)
         
         #past_key_values[:,:]
         input = torch.cat((input_ids,candidate),dim=-1)
         past_kv = repeat_kv(past_key_values,len(candidate))
         
         outputs = model.base_model.model(
                        input_ids=input,
                        attention_mask=attention_mask,
                        past_key_values=past_kv
                    )
         orig = model.base_model.lm_head(outputs[0])
         
         
         best_candidate_index = 0
         for i in range(len(candidate)):
             count = 0
             for j in range(totallen):
                 _,tk = orig[i][j].topk(k=topk , dim=-1)
                 
                 if sum(tk.eq(candidate[i][j])) : count = count+1
                 else: break
             if count > aclength: 
                 aclength = count
                 best_candidate = candidate[i][:aclength]
                 batch_index = i
             if aclength == totallen:
                 break
         ###拒绝后或者完成后重采样一个token
         newt = torch.argmax(orig[batch_index][aclength])
         new_kv = get_kv( outputs['past_key_values'],batch_index,aclength,totallen)
         best_candidate = torch.cat((best_candidate,newt.unsqueeze(0)),dim=-1).to(dtype=newt.dtype)
         last_hidden_states = outputs['last_hidden_state'][best_candidate_index][:aclength+1]
         return best_candidate , new_kv , last_hidden_states
    def temperature_predict(self,input_ids = None ,candidate = None ,choices =  3,candiate_prob =None,op = None,attention_mask=None,past_key_values= None,topk=1):
         """candidate 是所有可能token序列片段
         """
         aclength = 0
         totallen = len(candidate[0])
         
         best_candidate = torch.tensor([])
         
         input_ids = input_ids.repeat(len(candidate),1)
         #for i in candidate:
         batch_index = 0
         attention_mask =    attention_mask.repeat(len(candidate),1)
         
         #past_key_values[:,:]
         input = torch.cat((input_ids,candidate),dim=-1)
         past_kv = repeat_kv(past_key_values,len(candidate))
         
         outputs = self.base_model.model(
                        input_ids=input,
                        attention_mask=attention_mask,
                        past_key_values=past_kv
                    )
         logits = self.base_model.lm_head(outputs[0])
         orig = torch.softmax(logits,dim=-1)
         best_candidate_index = 0
         for i in range(len(candidate)):
             count = 0
             for j in range(totallen):
                 #_,tk = orig[i][j].topk(k=topk , dim=-1)
                 r = random.random()
                 if candiate_prob[i][j] <= 0 :count =count+1;continue
                 if orig[i][j][candidate[i][j]]/candiate_prob[i][j] < r : break
                 else : count =count+1
                 # if sum(tk.eq(candidate[i][j])) : count = count+1
             if count > aclength: 
                 aclength = count
                 best_candidate = candidate[i][:aclength]
                 batch_index = i
             if aclength == totallen:
                 break
         ###拒绝后或者完成后重采样一个token
         if aclength ==totallen:
             newt = torch.argmax(orig[batch_index][aclength],dim=-1)
         else:
             new_index = int(batch_index/(choices**(totallen-aclength)))
             repro = orig[batch_index][aclength] -op[aclength][new_index]
             repro[repro<0] = 0 
             repro = repro/repro.sum()
             newt = torch.multinomial(repro, 1, replacement=False)
             newt=newt[0]
         new_kv = get_kv( outputs['past_key_values'],batch_index,aclength,totallen)
         best_candidate = torch.cat((best_candidate,newt.unsqueeze(0)),dim=-1).to(dtype=newt.dtype)
         last_hidden_states = outputs['last_hidden_state'][best_candidate_index][:aclength+1]
         return best_candidate , new_kv , last_hidden_states
    def generate(self,inputs,max_length = 50,choices=3,max_predictlen = 4,topk=1,greedy_search=False):
        """
        Args:
        max_length 
        choices mean number of candidate token ( position n)
        
        """
        tokenizer = self.get_tokenizer()
        input = tokenizer([inputs])
        input_ids = torch.tensor(input.input_ids)
        attention_mask = torch.tensor(input.attention_mask)
        past_key_values = None
        output = input_ids 
        count = 0 
        last_hs = None
        fastbuffer =None
        alltoken = []
        chimera_buffers = {"buffer":None,"attention_mask":None , "position_ids": None,}
        outputs = self.base_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values
                )
        if last_hs is None : last_hs = outputs[0]
        else :last_hs = torch.cat((last_hs,outputs[0]),dim=-2)
        orig = self.base_model.lm_head(outputs[0])  
        past_key_values = outputs['past_key_values']
        input_ids = torch.argmax(orig[0][-1]).unsqueeze(0).unsqueeze(0)
        attention_mask = torch.cat((attention_mask,attention_mask[:,0].unsqueeze(0)),dim=-1)
        output = torch.cat((output,input_ids),dim=-1) 
        if  greedy_search == True:
            for i in range(max_length):
         
                #####回退使用无损加速策略,如果是top5就选择输出，就这样 
                
                candidate=self.headgenerate(input_ids=output,attention_mask=attention_mask,last_hs = last_hs,max_len =max_predictlen,
                                                   choices = choices)
                """evaluate candiate"""
                attention_mask_ev = torch.cat((attention_mask,attention_mask[:,0:max_predictlen]),dim=-1)
                best_candidate , newkv,new_last_hs= self.naive_predict(input_ids = input_ids[0],candidate = candidate[:,len(output[0]):] ,attention_mask=attention_mask_ev,past_key_values= past_key_values,topk=topk)
        
                """renew kv,atmk , input,output"""    
                #print("速度{}".format(len(best_candidate)))
                count +=len(best_candidate)
                if len(best_candidate) > 0:
                    output =  torch.cat((output,best_candidate.unsqueeze(0)),dim=-1)  
                    input_ids = best_candidate[-1].unsqueeze(0)#torch.cat((input_ids,best_candidate.unsqueeze(0)),dim=-1) 
                    attention_mask = torch.cat((attention_mask,attention_mask[:,0:len(best_candidate)]),dim=-1)
                    last_hs = torch.cat((last_hs,new_last_hs.unsqueeze(0)),dim=-2)
                    past_key_values = newkv
        
            ratio =      (count/max_length)*32/34   
            print("平均加速为",(count/max_length)*32/34)
            #print(tokenizer.decode(output[0]))
            return output ,ratio   
        else:
            for i in range(max_length):
         
                #####回退使用无损加速策略,如果是top5就选择输出，就这样 
                
                candidate,candiate_prob,op=self.headgenerate2(input_ids=output,attention_mask=attention_mask,last_hs = last_hs,max_len =max_predictlen ,
                                   choices = choices)
                """evaluate candiate"""
                attention_mask_ev = torch.cat((attention_mask,attention_mask[:,0:max_predictlen]),dim=-1)
                best_candidate , newkv,new_last_hs=  self.temperature_predict(input_ids = input_ids[0],candidate = candidate[:,len(output[0]):] ,choices=choices,candiate_prob=candiate_prob[:],op = op ,attention_mask=attention_mask_ev,past_key_values= past_key_values,topk=topk)
        
                """renew kv,atmk , input,output"""    
                #print("速度{}".format(len(best_candidate)))
                count +=len(best_candidate)
                if len(best_candidate) > 0:
                    output =  torch.cat((output,best_candidate.unsqueeze(0)),dim=-1)  
                    input_ids = best_candidate[-1].unsqueeze(0)#torch.cat((input_ids,best_candidate.unsqueeze(0)),dim=-1) 
                    attention_mask = torch.cat((attention_mask,attention_mask[:,0:len(best_candidate)]),dim=-1)
                    last_hs = torch.cat((last_hs,new_last_hs.unsqueeze(0)),dim=-2)
                    past_key_values = newkv
        
            ratio =      (count/max_length)*32/34   
            print("平均加速为",(count/max_length)*32/34)
            #print(tokenizer.decode(output[0]))
            return output ,ratio   
    