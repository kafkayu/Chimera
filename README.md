

# Chimera llm-interference speedup

# Chimera 



this respository is aimed at speeding up  llm interference.
# Chimera 




![speedup demo](/data/demo.gif)

## environment
python 3.10

transformers 4.31.0

pytorch 2.1.2

dataset 2.13.1

huggingface_hub     0.16.4 

fschat 0.2.28

## datasets
[Sharedgpt](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered)


## Chimera Weights
|Base Model|Chimera on hugging Face|
|-------|-------|
|Vicuna-7b-v1.3|[anonymous6690/Chimera-Vicuna-7b-v1.3](https://huggingface.co/anonymous6690/Chimera-Vicuna-7b-v1.3)|
|Vicuna-13b-v1.3|[anonymous6690/Chimera-Vicuna-13b-v1.3](https://huggingface.co/anonymous6690/Chimera-Vicuna-13b-v1.3)|
|LlaMA-2-7b|[anonymous6690/Chimera-LlaMA-2-7b](https://huggingface.co/anonymous6690/Chimera-LlaMA-2-7b)|
|LlaMA-2-13b|[anonymous6690/Chimera-LlaMA-2-13b](https://huggingface.co/anonymous6690/Chimera-LlaMA-2-13b)|


## model
supprt vicuna , llama-2  and mistral
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


## evaluate

```
cd ./chimera
python test.py model_path
```

## Acknowledge 
Our project is based on a lot of excellent work such as [Medusa](https://github.com/FasterDecoding/Medusa)  ,[vicuna](https://vicuna.lmsys.org/)









