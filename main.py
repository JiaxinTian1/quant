import os
import json
import yaml
from 


def load_quant_config(self, config_path: str) -> Dict:
        """从YAML文件加载量化配置"""
        with open(config_path, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"量化配置文件解析错误: {e}")


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
    # model_path = "/data00/models/DeepSeek-R1"
    model_path = "/data00/models/DeepSeek-R1-converted"
    # model_path = "/data01/models/DeepSeek-V3-5layer"
    quant_config_path = "./quant_configs/qwen_moe_config.yaml"
    quant_config_path = "./quant_configs/ds_config.yaml"
    save_path = "../quantized_model"
    save_path = "../quantized_model_ds"
    calib_path = "./datasets/calib.json"
    calib_path = "./datasets/test.json"



    # 3. 初始化量化服务
    quant_service = QuantService()
    
    # 4. 执行量化流程
    try:
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        quant_service.run_pipeline(
            model_path=model_path,
            save_path=save_path,
            quant_config_path=quant_config_path,
            calib_data=calib_data
        )
        print("量化流程执行成功！")
    except Exception as e:
        print(f"量化流程执行失败: {e}")
        raise

main()