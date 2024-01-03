# fastlayer-llm-interference

this respository is aimed at speeding up  llm interference

## environment
transformers

pytorch

dataset

## training
this is a sample
```
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1   ./medusa/Medusa/medusa/train/train.py --model_name_or_path ../model/vicuna-7b-v1.3 \
    --data_path ../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../data/ShareGPT_Vicuna_unfiltered/small_test.json" \
    --output_dir 3gram_4fastlayer_1227 \
    --num_train_epochs 1 \
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
    --medusa_num_heads 1 \
    --medusa_num_layers 1
```

## evaluate
--1.3 new code--
interference.ipynb is a easy evaluation for the fastlayer model.

input can be  any prompts , output is the accuracy of next_next token.

for example, 

input is "how are you? assitant:"

model will give 2 token,"I am"

we can calculate the prediction accuracy of the next_next token such as "am" in this example



## theory
our method is aimed at speeding up the interference of llm by means of fastlayer,which is a special structure to help the model get the n-gram. As we all know the most of llm is based on the atuoregressive model, we have to admit a fact that this kind  of structure caused the low efficiency of some special tasks ,such as Generative tasks.  
