import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch
import psutil
from sklearn.preprocessing import StandardScaler
import pickle

class SequenceCSVDataset(Dataset):
    def __init__(self, folder_path, seq_len=10, selected_columns=None, 
                 scaler=None, scaler_path=None, mode='train', train_ratio=0.8, val_ratio=0.1):
        """
        修復標準化洩漏的數據集類
        Args:
            mode: 'train', 'val', 'all' - 決定返回訓練集、驗證集或全部數據
            train_ratio: 訓練集比例，用於時間順序分割
            val_ratio: 驗證集比例，用於時間順序分割
        """
        self.seq_len = seq_len
        self.mode = mode
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.sequences = []
        self.file_indices = []  # 記錄每個序列屬於哪個檔案
        self.scaler = scaler
        self.scaler_path = scaler_path
        
        # 轉換為絕對路徑
        abs_folder_path = os.path.abspath(folder_path)
        self.files = sorted(glob.glob(os.path.join(abs_folder_path, "*.csv")))
        
        if not self.files:
            raise ValueError(f"找不到CSV文件在路徑: {abs_folder_path}")
        
        # 使用傳入的 selected_columns 或預設值
        if selected_columns is None:
            selected_columns = ['Consumption_Total', 'Generation_Total', 'Power_Demand']
        self.selected_columns = selected_columns
        
        # 載入並處理所有檔案數據
        self.file_data = []  # 儲存每個檔案的原始數據
        
        for file in self.files:
            df = pd.read_csv(file)
            
            # 檢查必要的列是否存在
            missing_cols = set(selected_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"文件 {file} 缺少列: {missing_cols}")
            
            # 選擇指定的欄位
            df_selected = df[selected_columns]
            
            # 處理缺失值 - 使用前向填充然後後向填充
            df_selected = df_selected.ffill().bfill()
            
            # 如果還有缺失值，使用均值填充
            df_selected = df_selected.fillna(df_selected.mean())
            
            data = df_selected.values.astype(np.float32)
            self.file_data.append(data)
        
        # 只在訓練模式或scaler為None時創建並fit標準化器
        if self.scaler is None:
            self.scaler = StandardScaler()
            if mode == 'train' or mode == 'all':
                # 只使用訓練數據fit標準化器
                train_data = self._get_train_data_for_scaler()
                self.scaler.fit(train_data)
                
                # 保存scaler
                if self.scaler_path:
                    with open(self.scaler_path, 'wb') as f:
                        pickle.dump(self.scaler, f)
                    print(f"Scaler已保存到: {self.scaler_path}")
            else:
                raise ValueError("驗證模式需要提供已訓練的scaler")
        
        # 創建序列
        self._create_sequences()
        
        print(f"模式: {mode}")
        print(f"成功載入 {len(self.sequences)} 個序列")
        print(f"每個序列形狀: {self.sequences[0].shape if self.sequences else 'N/A'}")
        print(f"使用的欄位: {selected_columns}")
        print(f"特徵維度: {len(selected_columns)}")
        print(f"檔案數量: {len(self.files)}")
        
        # 顯示內存使用
        process = psutil.Process(os.getpid())
        print(f"記憶體使用: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    
    def _get_train_data_for_scaler(self):
        """只使用訓練部分的數據來fit標準化器，避免標準化洩漏"""
        train_data_list = []
        
        for data in self.file_data:
            # 對每個檔案進行時間順序分割
            split_idx = int(len(data) * self.train_ratio)
            train_data = data[:split_idx]
            train_data_list.append(train_data)
        
        # 合併所有訓練數據
        return np.concatenate(train_data_list, axis=0)
    
    def _create_sequences(self):
        """根據模式創建對應的序列"""
        for file_idx, data in enumerate(self.file_data):
            # 標準化數據
            data_normalized = self.scaler.transform(data)
            
            # 根據模式決定使用數據的哪個部分
            if self.mode == 'train':
                # 只使用前train_ratio部分
                split_idx = int(len(data_normalized) * self.train_ratio)
                data_to_use = data_normalized[:split_idx]
            elif self.mode == 'val':
                # 只使用train_ratio到(train_ratio + val_ratio)部分
                train_end_idx = int(len(data_normalized) * self.train_ratio)
                val_end_idx = int(len(data_normalized) * (self.train_ratio + self.val_ratio))
                data_to_use = data_normalized[train_end_idx:val_end_idx]
            else:  # mode == 'all'
                data_to_use = data_normalized
            
            # 創建序列（確保有足夠的數據創建至少一個序列）
            if len(data_to_use) >= self.seq_len:
                for i in range(len(data_to_use) - self.seq_len + 1):
                    seq = data_to_use[i:i+self.seq_len]
                    self.sequences.append(seq)
                    self.file_indices.append(file_idx)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 返回 torch tensor
        sequence = self.sequences[idx]
        
        # 將序列切分為輸入和目標
        # 輸入：前 seq_len - 1 個時間步的所有特徵
        inputs = sequence[:-1, :]
        # 目標：最後一個時間步的最後一個特徵 (Power_Demand)
        target = sequence[-1, -1:]
        
        return torch.FloatTensor(inputs), torch.FloatTensor(target)
    
    def get_scaler(self):
        """返回標準化器，用於反標準化預測結果"""
        return self.scaler
    
    def get_file_info(self):
        """返回檔案資訊"""
        return {
            'files': self.files,
            'file_count': len(self.files),
            'sequences_per_file': [sum(1 for idx in self.file_indices if idx == i) 
                                 for i in range(len(self.files))]
        }

def create_datasets(folder_path, seq_len=97, selected_columns=None, 
                   train_ratio=0.8, val_ratio=0.1, scaler_path=None):
    """
    創建訓練和驗證數據集，避免標準化洩漏
    
    Returns:
        train_dataset, val_dataset, scaler
    """
    # 創建訓練數據集（會fit標準化器）
    train_dataset = SequenceCSVDataset(
        folder_path=folder_path,
        seq_len=seq_len,
        selected_columns=selected_columns,
        mode='train',
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        scaler_path=scaler_path
    )
    
    # 創建驗證數據集（使用已訓練的標準化器）
    val_dataset = SequenceCSVDataset(
        folder_path=folder_path,
        seq_len=seq_len,
        selected_columns=selected_columns,
        scaler=train_dataset.get_scaler(),
        mode='val',
        train_ratio=train_ratio,
        val_ratio=val_ratio
    )
    
    return train_dataset, val_dataset, train_dataset.get_scaler()
