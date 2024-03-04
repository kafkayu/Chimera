# Chimera llm-interference speedup

this respository is aimed at speeding up  llm interference.

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
```
cd ./chimera
python test.py model_path
```





## Reference
For technical details and full experimental results, please check the [paper](https://arxiv.org/abs/2402.15758)
```
@misc{zeng2024chimera,
      title={Chimera: A Lossless Decoding Method for Accelerating Large Language Models Inference by Fusing all Tokens}, 
      author={Ziqian Zeng and Jiahong Yu and Qianshi Pang and Zihao Wang and Huiping Zhuang and Cen Chen},
      year={2024},
      eprint={2402.15758},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
