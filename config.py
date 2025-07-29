import yaml
import os

class Config:
    """配置管理類"""
    def __init__(self, config_path=None):
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self._update_from_dict(config)
        else:
            # 默認配置
            self.data = {
                'folder_path': 'data/processed',
                'seq_len': 97,
                'selected_columns': ['Consumption_Total', 'Generation_Total', 'Power_Demand'],
                'scaler_path': 'scaler.pkl'
            }
            
            self.model = {
                'feature_dim': 3,
                'd_model': 256,
                'nhead': 8,
                'num_layers': 4,
                'output_dim': 1,
                'max_seq_length': 1000,
                'dropout': 0.1
            }
            
            self.training = {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 100,
                'early_stopping_patience': 10,
                'train_ratio': 0.8,
                'device': 'cuda',  # 'cuda', 'cpu', 'mps'
                'save_path': 'transformer_model.pth',
                'show_plot': True,
                'gradient_clip': 1.0,
                'weight_decay': 1e-5
            }
            
            self.logging = {
                'tensorboard': True,
                'log_dir': 'runs/transformer_experiment',
                'save_frequency': 5  # 每5個epoch保存一次
            }
    
    def _update_from_dict(self, config_dict):
        """從字典更新配置"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, value)
            else:
                setattr(self, key, value)
    
    def save(self, path):
        """保存配置到YAML文件"""
        config_dict = {
            'data': self.data,
            'model': self.model,
            'training': self.training,
            'logging': self.logging
        }
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
