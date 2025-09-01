
import os
import json

from transformers import AutoModelForCausalLM
from quant_tools.quant_service import QuantService
# from quant_tools.example import QuantService

def print_model_tensor_keys(model):
    """
    打印 from_pretrained 加载的模型中所有张量的 key。
    参数：
        model (nn.Module): 已加载的 PyTorch 模型实例
    """
    state_dict = model.state_dict()
    for key in state_dict.keys():
        print(key)




def main():
    """主函数示例"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 1. 创建示例配置文件（可选）
    # create_sample_config()
    
    # 2. 定义路径
    # model_path = "/data/models/Qwen3-8B"
    # model_path = "/data/models/Qwen3-8B"
    # model_path = "/data/models/Qwen3-30B-A3B-FP8"
    model_path = "/data00/models/Qwen3-235B-A22B-FP8"
    model_path = "/data00/models/DeepSeek-R1-converted"
    # model_path = "/data01/models/DeepSeek-V3-5layer"
    quant_config_path = "/data01/tjx/quant/quant_configs/config.yaml"
    quant_config_path = "/data01/tjx/quant/quant_configs/ds_config.yaml"
    save_dir = "../quantized_model_ds"
    calib_path = "/data01/tjx/quant/datasets/calib.json"
    calib_path = "/data01/tjx/quant/datasets/test.json"



    # 3. 初始化量化服务
    quant_service = QuantService()
    
    # 4. 执行量化流程
    try:
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        quant_service.run_pipeline(
            model_path=model_path,
            save_dir=save_dir,
            quant_config_path=quant_config_path,
            calib_data=calib_data
        )
        print("量化流程执行成功！")
    except Exception as e:
        print(f"量化流程执行失败: {e}")
        raise

main()