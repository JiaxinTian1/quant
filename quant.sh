#!/bin/bash 

# 1. 定义变量（注意：Bash 变量赋值不能有空格！`model_path=xxx` 而非 `model_path = xxx`）
# model_path="/data/models/Qwen3-8B"
# model_path="/data/models/Qwen3-8B"
model_path="/data00/models/Qwen3-8B"
model_path="/data00/models/Qwen3-235B-A22B-FP8"  
# model_path="/data00/models/DeepSeek-R1"
# model_path="/data00/models/DeepSeek-R1-converted"
# model_path="/data01/models/DeepSeek-V3-5layer"

quant_config_path="./quant_practice/configs/qwen3_moe/qwen3_moe_int4.yaml"  
# quant_config_path="./quant_practice/configs/deepseek_v3/deepseek_v3_int4.yaml"

save_path="../quantized_model"  
# save_path="../quantized_model_ds"

# calib_path="./quant_practice/datasets/default.json"
calib_path="./quant_practice/datasets/test.json"

# 2. 定义需要手动设置的参数（你之前未定义，需补充默认值或根据需求修改）
quant_type="int4" 
# quant_type="fp8_e4m3" 
quant_target="sglang"    

# 3. 执行 Python 脚本（关键：Bash 变量引用必须加 $，如 $model_path 而非 model_path）
python3 main.py \
    --model_path "$model_path" \
    --save_path "$save_path" \
    --quant_type "$quant_type" \
    --quant_target "$quant_target" \
    --calib_data_path "$calib_path"
    # --quant_config_path "$quant_config_path" \
    

# torchrun --nproc-per-node 8 main.py \
#     --model_path "$model_path" \
#     --save_path "$save_path" \
#     --quant_type "$quant_type" \
#     --quant_target "$quant_target" \
#     --quant_config_path "$quant_config_path" \
#     --calib_data_path "$calib_path"