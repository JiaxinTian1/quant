from quant_service.quant_service import QuantService, QuantParams
import argparse

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="量化服务入口脚本")
    
    # 添加需要的命令行参数
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="模型路径"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True, 
        help="保存路径"
    )
    parser.add_argument(
        "--quant_type", 
        type=str, 
        required=True, 
        help="量化类型，如 int4、fp8、int4-fp8 等"
    )
    parser.add_argument(
        "--quant_target", 
        type=str, 
        required=True, 
        help="量化目标，可选值: weight、activation、both"
    )
    parser.add_argument(
        "--quant_config_path", 
        type=str, 
        default=None,
        help="量化配置文件路径（可选），如 ./configs/quant.yaml"
    )
    parser.add_argument(
        "--calib_data_path", 
        type=str, 
        default=None,
        help="校准文件路径（可选）"
    )
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 将解析结果包装成dataclass
    quant_params = QuantParams(
        model_path=args.model_path,
        save_path=args.save_path,
        quant_type=args.quant_type,
        quant_target=args.quant_target,
        quant_config_path=args.quant_config_path,
        calib_data_path=args.calib_data_path
    )
    
    # 初始化量化服务并传入参数
    quant_service = QuantService()
    quant_service.quant_params = quant_params
    quant_service.run_pipeline()
    

if __name__ == "__main__":
    main()