import os
import yaml
import json
from typing import Dict, List


def get_default_config_practice(model_type: str, quant_type: str) -> Dict:
    """
    拼接量化配置文件路径并读取YAML内容
    
    参数:
        model_type: 模型类型（如 "qwen3"、"deepseek"）
        quant_type: 量化类型（如 "int4"、"fp8"）
    
    返回:
        dict: YAML文件中的配置字典；若文件不存在则抛出FileNotFoundError
    """
    # 拼接配置文件路径：quant_config/{model_type}_{quant_type}.yaml
    # 例如：quant_config/qwen3_int4-fp8.yaml
    relative_parts = [
        "configs",       # 根配置目录
        model_type,           # 模型类型子目录
        f"{model_type}_{quant_type}.yaml"  # 文件名
    ]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, *relative_parts)
    config_path = os.path.normpath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
            return config_dict if config_dict is not None else {}
        except yaml.YAMLError as e:
            raise ValueError(f"解析YAML文件失败: {e}")


def get_default_data_practice() -> List:
    """
    拼接默认数据文件路径并读取JSON内容

    返回:
        List: JSON文件中的数据
    
    异常:
        FileNotFoundError: 数据文件不存在时抛出
        ValueError: JSON解析失败或文件读取错误时抛出
    """
    relative_parts = [
        "datasets",               # 数据根目录
        "default.json"   # 数据文件名（如 calib.json、test.json）
    ]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, *relative_parts)
    data_path = os.path.normpath(data_path)
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except FileNotFoundError:
        raise FileNotFoundError(f"默认数据文件不存在: {data_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON数据解析失败({data_path}): {str(e)}")
    except IOError as e:
        raise ValueError(f"数据文件读取错误（{data_path}): {str(e)}")