import torch
from src.DataLoader import create_datasets
from src.Model import TransformerModel
from src.Trainer import TransformerTrainer
from config import Config
import argparse
import os

def main():
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='Train Transformer Model')
    parser.add_argument('--config', type=str, default='config.yaml', 
                       help='配置文件路徑')
    parser.add_argument('--resume', type=str, default=None,
                       help='從檢查點恢復訓練')
    args = parser.parse_args()
    
    # 加載配置
    config = Config(args.config)
    
    # 設置隨機種子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 創建數據集（避免標準化洩漏）
    print("載入數據集...")
    train_dataset, val_dataset, _ = create_datasets(
        folder_path=config.data['folder_path'], 
        seq_len=config.data['seq_len'],
        selected_columns=config.data['selected_columns'],
        train_ratio=config.training['train_ratio'],
        scaler_path=config.data['scaler_path']
    )
    
    # 創建模型
    print("創建模型...")
    model = TransformerModel(**config.model)
    
    # 創建訓練器
    trainer = TransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config.training['batch_size'],
        learning_rate=config.training['learning_rate'],
        device=config.training['device'],
        show_plot=config.training['show_plot'],
        config=config
    )
    
    # 如果需要恢復訓練
    start_epoch = 0
    if args.resume:
        checkpoint = trainer.load_model(args.resume)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
    
    # 開始訓練
    print("開始訓練...")
    trainer.train(
        num_epochs=config.training['num_epochs'],
        save_path=config.training['save_path'],
        early_stopping_patience=config.training['early_stopping_patience'],
        start_epoch=start_epoch
    )
    
    # 繪製訓練總結
    trainer.plot_summary()
    
    # 保存最終配置
    config.save('final_config.yaml')
    print("訓練完成！")

if __name__ == "__main__":
    main()
