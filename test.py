import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml
import pickle
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ========== 複製 Config 類 ==========
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
                'selected_columns': ['AC1', 'AC2', 'AC3', 'AC4', 'Dish washer', 'Washing Machine', 'Dryer', 'Water heater', 'TV', 'Microwave', 'Kettle', 'Lighting', 'Refrigerator', 'Consumption_Total', 'Generation_Total', 'TemperatureC', 'DewpointC', 'PressurehPa', 'WindSpeedKMH', 'WindSpeedGustKMH', 'Humidity', 'HourlyPrecipMM', 'dailyrainMM', 'SolarRadiationWatts_m2', 'Power_Demand'],
                'scaler_path': 'scaler.pkl'
            }
            
            self.model = {
                'feature_dim': 25,
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
                'val_ratio': 0.1,
                'device': 'cuda',
                'save_path': 'transformer_model.pth',
                'show_plot': True,
                'gradient_clip': 1.0,
                'weight_decay': 1e-5
            }
            
            self.logging = {
                'tensorboard': True,
                'log_dir': 'runs/transformer_experiment',
                'save_frequency': 5
            }
    
    def _update_from_dict(self, config_dict):
        """從字典更新配置"""
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, value)
            else:
                setattr(self, key, value)

# ========== 複製 Model 類 ==========
class PositionalEncoder(nn.Module):
    """位置編碼器：為序列中的每個位置添加位置信息"""
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    """改進的Transformer模型，包含殘差連接和層正規化"""
    def __init__(self, feature_dim, d_model=512, nhead=8, num_layers=6, 
                 output_dim=None, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        self.pos_encoder = PositionalEncoder(d_model, max_seq_length)
        
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        self.attention_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        if output_dim is not None:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )
        else:
            self.output_proj = None
            
        self.init_weights()
    
    def init_weights(self):
        """使用Xavier初始化權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _encode(self, x, src_mask=None):
        """執行從輸入到Transformer編碼器的過程"""
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x * math.sqrt(self.d_model)

        x = self.pos_encoder(x)
        x = self.dropout(x)

        output = self.transformer(x, src_mask)
        return output

    def forward(self, x, src_mask=None):
        output = self._encode(x, src_mask)

        attention_scores = self.attention_proj(output)
        attention_weights = torch.softmax(attention_scores, dim=1)

        weighted_output = (attention_weights * output).sum(dim=1)

        if self.output_proj is not None:
            final_output = self.output_proj(weighted_output)
        else:
            final_output = weighted_output

        return final_output

    def get_attention_weights(self, x, src_mask=None):
        """獲取注意力權重用於可視化"""
        output = self._encode(x, src_mask)

        attention_scores = self.attention_proj(output)
        attention_weights = torch.softmax(attention_scores, dim=1)

        return attention_weights

# ========== 建立單檔案資料集類別 ==========
class SingleFileTestDataset(Dataset):
    """用於測試單個檔案最後10%資料的資料集類別"""
    def __init__(self, file_path, seq_len=97, selected_columns=None, scaler=None):
        self.seq_len = seq_len
        self.sequences = []
        self.scaler = scaler
        
        # 使用傳入的 selected_columns 或預設值
        if selected_columns is None:
            selected_columns = ['Consumption_Total', 'Generation_Total', 'Power_Demand']
        self.selected_columns = selected_columns
        
        # 載入檔案
        df = pd.read_csv(file_path)
        
        # 檢查必要的列是否存在
        missing_cols = set(selected_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"文件 {file_path} 缺少列: {missing_cols}")
        
        # 選擇指定的欄位
        df_selected = df[selected_columns]
        
        # 處理缺失值
        df_selected = df_selected.ffill().bfill()
        df_selected = df_selected.fillna(df_selected.mean())
        
        data = df_selected.values.astype(np.float32)
        
        # 只使用最後10%的資料（相當於驗證集）
        split_idx = int(len(data) * 0.8)  # 前80%是訓練集
        data_to_use = data[split_idx:]  # 後20%是驗證集，但我們只要最後10%
        
        # 再取後半部分作為最後10%
        final_split = int(len(data_to_use) * 0.5)
        data_final = data_to_use[final_split:]
        
        # 標準化資料
        if self.scaler is not None:
            data_normalized = self.scaler.transform(data_final)
        else:
            raise ValueError("需要提供已訓練的scaler")
        
        # 創建序列
        if len(data_normalized) >= self.seq_len:
            for i in range(len(data_normalized) - self.seq_len + 1):
                seq = data_normalized[i:i+self.seq_len]
                self.sequences.append(seq)
        
        print(f"載入檔案: {os.path.basename(file_path)}")
        print(f"原始資料長度: {len(data)}")
        print(f"最後10%資料長度: {len(data_final)}")
        print(f"創建序列數: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # 將序列切分為輸入和目標
        inputs = sequence[:-1, :]  # 前 seq_len - 1 個時間步的所有特徵
        target = sequence[-1, -1:]  # 最後一個時間步的最後一個特徵 (Power_Demand)
        
        return torch.FloatTensor(inputs), torch.FloatTensor(target)

# ========== 圖表生成函數 ==========
def plot_predictions(model, data_loader, device, save_path, show_plot=False):
    """繪製預測結果與實際值折線圖並儲存"""
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            predictions = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(all_targets, label='Actual Power Demand', color='blue')
    plt.plot(all_predictions, label='Predicted Power Demand', color='orange')
    plt.title('Test Set: Predicted vs. Actual Power Demand')
    plt.xlabel('Sample Index')
    plt.ylabel('Power_Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def plot_perfect_prediction(model, data_loader, device, save_path, show_plot=False):
    """繪製完美預測線與預測值點圖並儲存"""
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            predictions = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    plt.figure(figsize=(6, 6))
    plt.scatter(all_targets, all_predictions, alpha=0.5, color='green', label='Predicted Points')
    min_val = min(all_targets.min(), all_predictions.min())
    max_val = max(all_targets.max(), all_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction Line')
    plt.title('Predicted vs. Actual (Perfect Prediction Line)')
    plt.xlabel('Actual Power Demand')
    plt.ylabel('Predicted Power Demand')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

def plot_error_percentage_summary(model, data_loader, device, save_dir, show_plot=False):
    """繪製實際值與預測值的百分比誤差折線圖和直方圖"""
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            predictions = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    # 計算誤差百分比
    error_percentage = ((all_predictions - all_targets) / (all_targets + 1e-8)) * 100

    # 1.折線圖
    plt.figure(figsize=(12, 6))
    plt.plot(error_percentage, color='purple', alpha=0.6, label='Error Percentage')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Prediction Error Percentage on Test Set')
    plt.xlabel('Sample Index')
    plt.ylabel('Error (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_percentage_line.png'))
    if show_plot:
        plt.show()
    plt.close()

    # 2.直方圖
    plt.figure(figsize=(8, 6))
    plt.hist(error_percentage, bins=100, color='orange', alpha=0.7, edgecolor='black')
    plt.title('Prediction Error Percentage Distribution')
    plt.xlabel('Error (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_percentage_histogram.png'))
    if show_plot:
        plt.show()
    plt.close()

    # 打印誤差統計
    print(f"Error Percentage Statistics:")
    print(f" Mean: {error_percentage.mean():.2f}%")
    print(f" Std:  {error_percentage.std():.2f}%")
    print(f" Max:  {error_percentage.max():.2f}%")
    print(f" Min:  {error_percentage.min():.2f}%")

def plot_attention_weights(model, data_loader, device, save_dir, show_plot=False):
    """可視化注意力權重，顯示模型關注的時間步"""
    model.eval()
    
    # 取一個批次的數據來分析
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.float().to(device)
            
            # 獲取注意力權重
            attention_weights = model.get_attention_weights(inputs)
            attention_weights = attention_weights.squeeze(-1).cpu().numpy()
            
            # 只取前5個樣本進行可視化
            num_samples = min(5, attention_weights.shape[0])
            
            plt.figure(figsize=(15, 3 * num_samples))
            for i in range(num_samples):
                plt.subplot(num_samples, 1, i + 1)
                weights = attention_weights[i]
                
                # 繪製注意力權重
                plt.plot(range(len(weights)), weights, 'b-', linewidth=2)
                plt.fill_between(range(len(weights)), weights, alpha=0.3)
                plt.title(f'Sample {i+1}: Attention Weights Across Time Steps')
                plt.xlabel('Time Step')
                plt.ylabel('Attention Weight')
                plt.grid(True, alpha=0.3)
                
                # 標記最高權重的時間步
                max_idx = np.argmax(weights)
                plt.scatter(max_idx, weights[max_idx], color='red', s=100, zorder=5)
                plt.text(max_idx, weights[max_idx], f'Max: {weights[max_idx]:.3f}', 
                        verticalalignment='bottom', horizontalalignment='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'attention_weights.png'))
            if show_plot:
                plt.show()
            plt.close()
            
            # 計算平均注意力權重
            avg_weights = attention_weights.mean(axis=0)
            plt.figure(figsize=(12, 4))
            plt.plot(range(len(avg_weights)), avg_weights, 'r-', linewidth=3)
            plt.fill_between(range(len(avg_weights)), avg_weights, alpha=0.3, color='red')
            plt.title('Average Attention Weights Across All Test Samples')
            plt.xlabel('Time Step')
            plt.ylabel('Average Attention Weight')
            plt.grid(True, alpha=0.3)
            
            # 標記最重要的時間步
            top_5_indices = np.argsort(avg_weights)[-5:]
            for idx in top_5_indices:
                plt.scatter(idx, avg_weights[idx], color='blue', s=60, zorder=5)
                plt.text(idx, avg_weights[idx], f'{idx}', 
                        verticalalignment='bottom', horizontalalignment='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'average_attention_weights.png'))
            if show_plot:
                plt.show()
            plt.close()
            
            print(f"注意力權重統計:")
            print(f"最重要的5個時間步: {top_5_indices}")
            print(f"對應的平均權重: {avg_weights[top_5_indices]}")
            
            break  # 只處理第一個批次

def sMAPE(y_true, y_pred):
    """計算sMAPE"""
    if y_true == 0 and y_pred == 0:
        return 0
    return 2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)) * 100

def plot_sMAPE_summary(model, data_loader, device, save_dir, show_plot=False):
    """繪製sMAPE圖表"""
    model.eval()
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            predictions = model(inputs)
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0).flatten()
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()

    # calculate sMAPE for each sample
    sMAPE_values = []
    for i in range(len(all_targets)):
        smape_val = sMAPE(all_targets[i], all_predictions[i])
        sMAPE_values.append(smape_val)
    
    sMAPE_values = np.array(sMAPE_values)

    # 1.折線圖
    plt.figure(figsize=(12, 6))
    plt.plot(sMAPE_values, color='purple', alpha=0.6, label='sMAPE')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Prediction sMAPE on Test Set')
    plt.xlabel('Sample Index')
    plt.ylabel('sMAPE (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sMAPE_line.png'))
    if show_plot:
        plt.show()
    plt.close()

    # 2.直方圖
    plt.figure(figsize=(8, 6))
    plt.hist(sMAPE_values, bins=100, color='orange', alpha=0.7, edgecolor='black')
    plt.title('Prediction sMAPE Distribution')
    plt.xlabel('sMAPE (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sMAPE_histogram.png'))
    if show_plot:
        plt.show()
    plt.close()

    # print sMAPE statistics
    print(f"sMAPE Statistics:")
    print(f" Mean: {sMAPE_values.mean():.2f}%")
    print(f" Std:  {sMAPE_values.std():.2f}%")
    print(f" Max:  {sMAPE_values.max():.2f}%")
    print(f" Min:  {sMAPE_values.min():.2f}%")

def evaluate_single_file(file_path, model, config, scaler, device, results_dir):
    """對單個檔案進行推論並生成所有圖表"""
    print(f"\n處理檔案: {os.path.basename(file_path)}")
    
    # 建立結果資料夾
    file_name = os.path.basename(file_path).replace('.csv', '')
    save_dir = os.path.join(results_dir, file_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # 建立資料集
    try:
        dataset = SingleFileTestDataset(
            file_path=file_path,
            seq_len=config.data['seq_len'],
            selected_columns=config.data['selected_columns'],
            scaler=scaler
        )
        
        if len(dataset) == 0:
            print(f"警告: {file_name} 沒有足夠的資料創建序列，跳過此檔案")
            return
        
        # 建立資料載入器
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
        
        # 計算並打印測試損失
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs = inputs.float().to(device)
                targets = targets.float().to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"測試損失 (MSE): {avg_loss:.6f}")
        
        # 生成所有圖表
        print("生成預測vs實際值圖表...")
        plot_predictions(model, data_loader, device, 
                        os.path.join(save_dir, 'predictions.png'), show_plot=False)
        
        print("生成完美預測線圖表...")
        plot_perfect_prediction(model, data_loader, device, 
                              os.path.join(save_dir, 'perfect_prediction.png'), show_plot=False)
        
        print("生成誤差百分比圖表...")
        plot_error_percentage_summary(model, data_loader, device, save_dir, show_plot=False)
        
        print("生成注意力權重圖表...")
        plot_attention_weights(model, data_loader, device, save_dir, show_plot=False)

        print("生成sMAPE圖表...")
        plot_sMAPE_summary(model, data_loader, device, save_dir, show_plot=False)
        
        print(f"✓ {file_name} 處理完成，結果儲存至 {save_dir}")
        
    except Exception as e:
        print(f"處理 {file_name} 時發生錯誤: {str(e)}")

def main():
    """主函數"""
    print("開始測試推論...")
    
    # 載入配置
    config = Config('final_config.yaml')
    
    # 設置設備 - 強制使用CPU以確保穩定運行
    device = torch.device('mps')
    print(f"使用設備: {device}")
    
    # 載入標準化器
    with open(config.data['scaler_path'], 'rb') as f:
        scaler = pickle.load(f)
    print("標準化器載入成功")
    
    # 創建模型
    model = TransformerModel(**config.model)
    model.to(device)
    
    # 載入訓練好的模型權重
    checkpoint = torch.load('transformer_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("模型載入成功")
    
    # 建立結果資料夾
    results_dir = 'results_local'
    os.makedirs(results_dir, exist_ok=True)
    
    # 取得所有要處理的檔案
    data_folder = config.data['folder_path']
    consumer_files = sorted(glob.glob(os.path.join(data_folder, "Consumer_*.csv")))
    public_building_file = os.path.join(data_folder, "Public_Building.csv")
    
    all_files = consumer_files.copy()
    if os.path.exists(public_building_file):
        all_files.append(public_building_file)
    
    print(f"找到 {len(all_files)} 個檔案需要處理")
    print(f"Consumer 檔案: {len(consumer_files)} 個")
    print(f"Public_Building 檔案: {'存在' if os.path.exists(public_building_file) else '不存在'}")
    
    # 處理每個檔案
    for i, file_path in enumerate(all_files, 1):
        print(f"\n[{i}/{len(all_files)}] ", end="")
        evaluate_single_file(file_path, model, config, scaler, device, results_dir)
    
    print(f"\n所有檔案處理完成！結果儲存在 {results_dir} 資料夾中")

if __name__ == "__main__":
    main()