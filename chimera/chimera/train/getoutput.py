import transformers
from transformers import Trainer, BitsAndBytesConfig
import torch
import json
#####1.加载所有ques
filename = "../../../../../data/ShareGPT_Vicuna_unfiltered/1280question.json" #问题数据
model_name_or_path = "../../../../../model/vicuna-7b-v1.3" #####模型路径


with open( filename, "r") as json_file:
    Q = json.load(json_file)
####2.加载7b模型

load_in_4bit = True
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config if load_in_4bit else None,
        load_in_4bit=load_in_4bit,
    )
model_max_length=2048
tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
tokenizer.pad_token = tokenizer.unk_token




#######3.generate data
save_filename =  "../../../../../data/ShareGPT_Vicuna_unfiltered/1280question_TEST.json"
for i in range(len(Q['question'])):
    if i%10==0 : print("进度{}".format(i/len(Q['question'])))
    if i % 50 == 0:
        with open( filename, "w") as json_file:
            json.dump(Q,json_file)
        print("已经保存{}个question".format(i))
    input = tokenizer([Q['question'][i]['value']])
    max_length = min(len(input['input_ids'][0])*10,2048)
    if len(input['input_ids'][0]) < 2048:
        output = model.generate(inputs = torch.tensor(input['input_ids'] ).to("cuda"),max_length = len(input['input_ids'][0])+max_length,early_stopping =True)
        text = tokenizer.decode(output[0],skip_special_tokens =True)
        Q['question'][i]['answer'] = text
    torch.cuda.empty_cache()
print("计算完毕")
with open( save_filename, "w") as json_file:
            json.dump(Q,json_file)