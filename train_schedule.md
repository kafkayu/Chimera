1.5 实验
13b测试
1.5 - 1.7
下载13b模型
https://huggingface.co/lmsys/vicuna-13b-v1.5

```
torchrun --nproc_per_node=1   medusa/train/train.py --model_name_or_path ../../../model/vicuna-13b-v1.3 \
    --data_path ../../../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../../../data/ShareGPT_Vicuna_unfiltered/test.json" \
    --output_dir idea9_3gram_0106 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 1 \
    --medusa_num_layers 1
```
1.6 

| 任务| 负责人 | dll说明 | 详细说明|
|-------|-------|-------|-------|
| 跑7b多token结果 | 俞嘉鸿 | 1.6 20：00 结束| 得知预测多个token的准确率|
| 13b结果测试 | zihao-wang | 等待nvidia回复后 |得知预测多个token的准确率|
| eagle baseline 测试 | 庞千石 | 1.7晚上19：00 |查看eagle 1b,7b,13b的加速效果，eagle 预测多个token的准确率|



2.若服务器可运行，13b结果测试 

下载13b模型 
https://huggingface.co/lmsys/vicuna-13b-v1.5

```
torchrun --nproc_per_node=1   medusa/train/train.py --model_name_or_path ../../../model/vicuna-13b-v1.3 \
    --data_path ../../../data/ShareGPT_Vicuna_unfiltered/train.json \
    --eval_data_path  "../../../data/ShareGPT_Vicuna_unfiltered/test.json" \
    --output_dir idea6_3gram_0106 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "steps" \
    --eval_steps 50 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --medusa_num_heads 1 \
    --medusa_num_layers 1
```
3.eagle baseline 测试 负责人：庞千石 ddl:1.7晚上19：00

[eagle](https://github.com/SafeAILab/EAGLE) 地址

期望结果：
查看eagle 1b,7b,13b的加速效果
eagle top5 token的预测准确率

1.8 
| 任务| 负责人 | dll说明 | 详细说明|
|-------|-------|-------|-------|
| 跑7b多token结果 | 俞嘉鸿 | 1.9 24：00 结束| 已经确定好模型架构，现在需要编写多个token预测加速代码，并且需要再次训练7b结构模型|
| eagle baseline 测试 | 庞千石 | 1.9晚上19：00 |llama2-70b-chat据项目数据的mt_bench加速效果检测为3x，7b的mt_bench加速效果为2.3左右，查看eagle 7b,13b的加速效果，eagle 预测多个token的准确率|

1.9
| 任务| 负责人 | dll说明 | 详细说明| 
|-------|-------|-------|-------|
|1.模型加速推理验证代码 | 俞嘉鸿 | 1.10 24：00 结束| 已经确定好模型架构，现在需要编写多个token预测加速代码|
|2. lookahead baseline 测试 | 庞千石 | 1.11晚上19：00 |[lookahead项目链接](https://lmsys.org/blog/2023-11-21-lookahead-decoding/?utm_source=talkingdev.uwl.me)|
|3. vicuna-7b数据集 生成| 庞千石 | 1.11晚上19：00 |使用vicuna-7b，根据question，max_length设置为2048，greedysearch产生一批新的模型输出|
|4.模型量化 | 俞嘉鸿 | 1.12 24：00 结束| 要求:13b模型本体量化使其可以在32G或者48G显存的gpu上进行运行|
|5.早期退出验证 | 俞嘉鸿 | 1.12 24：00 结束| 要求:13b模型本体量化使其可以在32G或者48G显存的gpu上进行运行|
|6.trigram消融 | 俞嘉鸿 | 1.16 24：00 结束| 要求:去掉trigram后是否会影响性能，影响如何|
|7.13b模型验证效果 | zihaowang | 1.16 24：00 结束| 要求:在量化实现可以在32G装下模型后，使其可以train13b模型|

1.11
| 任务| 负责人 | dll说明 | 详细说明| 进度|
|-------|-------|-------|-------|-------|
|1.模型加速推理验证代码 | 俞嘉鸿 | 1.11 24：00 结束| 已经确定好模型架构，现在需要编写多个token预测加速代码|编写了greedy search 代码，加速大约为2.5x，编写beamsearch代码中|
|2. lookahead baseline 测试 | 庞千石 | 1.14晚上19：00 |[lookahead项目链接](https://lmsys.org/blog/2023-11-21-lookahead-decoding/?utm_source=talkingdev.uwl.me)|llama2-7b-chat-hf 在mt_bench上加速约1.6x||
|3. vicuna-7b数据集 生成| 庞千石 | 1.14晚上19：00 |使用vicuna-7b，根据question，max_length设置为512，greedysearch产生一批新的模型输出|程序已在运行中，预计1月26号将完成所有问题的推理(其中每一百个问题需要约35分钟)、已完成，获9G问答文件|
|4.模型量化 | 俞嘉鸿 | 1.12 24：00 结束| 要求:13b模型本体量化使其可以在32G或者48G显存的gpu上进行运行|修改模型并进行量化|正在尝试将模型进行量化，量化为int8减少显存占用|
|5.早期退出验证 | 俞嘉鸿 | 1.12 24：00 结束| 要求:13b模型本体量化使其可以在32G或者48G显存的gpu上进行运行||
|6.trigram消融 | 俞嘉鸿 | 1.16 24：00 结束| 要求:去掉trigram后是否会影响性能，影响如何|去掉trigram 后top5 acc大约降到了0.87，原来模型有0.9左右，下降了三个点|
|7.13b模型验证效果 | zihaowang | 1.16 24：00 结束| 要求:在量化实现可以在32G装下模型后，使其可以train13b模型||

