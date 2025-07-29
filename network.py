import netron
import torch
import os
import argparse
from src.Model import TransformerModel
from config import Config

def visualize_transformer_model(model_path, config_path):
    """
    使用netron可视化Transformer模型
    """
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"錯誤：找不到配置文件 {config_path}")
        print("請確保與模型匹配的config.yaml文件存在。")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"錯誤：找不到模型文件 {model_path}")
        return
    
    print("正在加載模型...")
    
    try:
        # 加载模型状态字典
        checkpoint = torch.load(model_path, map_location='cpu')

        # 從配置文件加載模型參數
        print(f"正在從 {config_path} 加載配置...")
        config = Config(config_path)
        
        # 根據配置創建模型實例
        model = TransformerModel(**config.model)
        
        # 加載模型權重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("模型加載成功！")
        
        # 將模型設為評估模式
        model.eval()
        
        # 創建示例輸入用於可視化
        # 輸入形狀: (batch_size, seq_len, feature_dim)
        sample_input = torch.randn(1, config.data['seq_len'] - 1, 
                                   config.model['feature_dim'])
        
        # 導出為ONNX格式以便netron可視化
        onnx_path = "transformer_model.onnx"
        print("正在導出ONNX格式...")
        
        torch.onnx.export(
            model,                          # 模型
            sample_input,                   # 示例輸入
            onnx_path,                      # 輸出文件路徑
            export_params=True,             # 導出參數
            opset_version=14,               # ONNX操作集版本（修復unflatten支持問題）
            do_constant_folding=True,       # 常數折疊優化
            input_names=['input'],          # 輸入名稱
            output_names=['output'],        # 輸出名稱
            dynamic_axes={                  # 動態軸
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"ONNX模型已保存到: {onnx_path}")
        
        # 使用netron可視化ONNX模型
        print("正在啟動netron可視化...")
        print("瀏覽器將會自動打開，顯示模型結構圖")
        print("按 Ctrl+C 停止服務器")
        
        # 啟動netron服務器
        netron.start(onnx_path, browse=True)
        
    except Exception as e:
        print(f"載入模型時發生錯誤: {str(e)}")
        print("請確保模型文件格式正確，且與模型定義匹配")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Transformer Model")
    parser.add_argument('--model', type=str, default='transformer_model.pth',
                        help='Path to the PyTorch model file (.pth)')
    parser.add_argument('--config', type=str, default='final_config.yaml',
                        help='Path to the model config file (.yaml)')
    args = parser.parse_args()

    print("=== Transformer模型可視化 ===")
    print("選項 1: 可視化ONNX格式（推薦，顯示完整結構）")
    print("選項 2: 直接可視化PyTorch文件")
    
    choice = input("請選擇 (1/2, 默認為1): ").strip()

    if choice == "2":
        # 直接可視化PyTorch模型
        print("正在啟動netron（PyTorch格式）...")
        netron.start(args.model, browse=True)
    else:
        # 可視化ONNX格式（推薦）
        visualize_transformer_model(args.model, args.config)