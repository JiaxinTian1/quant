# 量化工具

使用案例

```python

import os
from quant_tools.quant_service import QuantService


def main():
    """主函数示例"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # 1. 创建示例配置文件（可选）
    # create_sample_config()
    
    # 2. 定义路径
    model_path = "/data/models/Qwen3-8B"
    quant_config_path = "/data/tjx/my_tool/quant_config/config.yaml"
    save_dir = "./quantized_model"

    # 3. 初始化量化服务
    quant_service = QuantService()
    
    # 4. 执行量化流程
    try:
        quant_service.run_pipeline(
            model_path=model_path,
            save_dir=save_dir,
            quant_config_path=quant_config_path,
            calib_data=[
                "这是一个量化校准样本",
                "自然语言处理中的模型量化技术", 
                "Transformer模型结构包含自注意力机制",
                "量化可以显著减少模型大小并加速推理",
                "深度学习模型的压缩是当前研究的热点"
            ]
        )
        print("量化流程执行成功！")
    except Exception as e:
        print(f"量化流程执行失败: {e}")
        raise

main()

```
