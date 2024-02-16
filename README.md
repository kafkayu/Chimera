# Chimera 

this respository is aimed at speeding up  llm interference.

## environment
python 3.10

transformers 4.31.0

pytorch 2.1.2

dataset 2.13.1

huggingface_hub     0.16.4 

fschat 0.2.28

## datasets
Sharedgpt


## Chimera Weights
|Base Model|Chimera on hugging Face|
|-------|-------|
|Vicuna-7b-v1.3|([anonymous6690/Chimera-Vicuna-7b-v1.3]https://huggingface.co/anonymous6690/Chimera-Vicuna-7b-v1.3)|
|Vicuna-13b-v1.3|([anonymous6690/Chimera-Vicuna-13b-v1.3]https://huggingface.co/anonymous6690/Chimera-Vicuna-13b-v1.3)|
|LlaMA-2-7b|([anonymous6690/Chimera-LlaMA-2-7b]https://huggingface.co/anonymous6690/Chimera-LlaMA-2-7b)|
|LlaMA-2-13b|([anonymous6690/Chimera-LlaMA-2-13b]https://huggingface.co/anonymous6690/Chimera-LlaMA-2-13b)|


## model
vicuna7b/13b
```
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-v1.5
```
```
git lfs install
git clone https://huggingface.co/lmsys/vicuna-13b-v1.5
```

## training

### easy example
this is a sample

if you want to use wandb to watch the accuracy of prediction, please use your own wandb key and change the wandb.init("**your key**") in the train.py

```
cd ./chimera
torchrun --nproc_per_node=1   ./train.py --model_name_or_path ../model/vicuna-7b-v1.3 \
    --data_path ../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../data/ShareGPT_Vicuna_unfiltered/small_test.json" \
    --output_dir chimera_0125 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --chimera_num_heads 4 \
    --chimera_num_layers 1
```
### quantinization support 

you can set  the parameters of **--load_in_4bit** or **--load_in_8bit** in the model quantinization

It should be awared that the version of transformers must compatible with our requirement , you may need to change the init code of chimera model instead.
```
cd ./chimera
torchrun --nproc_per_node=1  ./chimera/train.py --model_name_or_path ../model/vicuna-7b-v1.3 --load_in_4bit True\
    --data_path ../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../data/ShareGPT_Vicuna_unfiltered/small_test3.json" \
    --output_dir chimera_1f_posthalf_finetune\
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2\
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 50\
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 1024 \
    --lazy_preprocess True \
    --chimera_num_heads 4 \
    --chimera_num_layers 1
```

## evaluate
--1.3 new code--
medusa/train/interference.ipynb is an easy evaluation for the fastlayer model.

input can be  any prompts , output is the accuracy of next_next token.

for example, 

input is "how are you? assitant:"

model will give 2 token,"I am"

we can calculate the prediction accuracy of the next_next token such as "am" in this example



## theory
our method is aimed at speeding up the interference of llm by means of fastlayer,which is a special structure to help the model get the n-gram. As we all know the most of llm is based on the atuoregressive model, we have to admit a fact that this kind  of structure caused the low efficiency of some special tasks ,such as Generative tasks.  Speculative decoding is a good idea by making good use of the paralism of llm.However , there is a still difficult problem how we can get the n-gram with the minist cost.In the fastlayer model,we use the trigram and extended transformer to assit the big llm to predict more token with the least cost.Meanwhile,we provide a new method to evaluate the speculative token which can meet the demand of greedy search and beam search. 
