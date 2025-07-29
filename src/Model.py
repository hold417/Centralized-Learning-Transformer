import torch
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    """
    位置編碼器：為序列中的每個位置添加位置信息
    """
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # 創建位置編碼矩陣
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # 計算div_term用於正弦和餘弦函數
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        # 偶數索引使用sin，奇數索引使用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 註冊為buffer，不會被視為模型參數
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_seq_length, d_model)
    
    def forward(self, x):
        """
        x: (batch_size, seq_length, d_model)
        """
        # 添加位置編碼
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    """
    改進的Transformer模型，包含殘差連接和層正規化
    """
    def __init__(self, feature_dim, d_model=512, nhead=8, num_layers=6, 
                 output_dim=None, max_seq_length=5000, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # 輸入投影層 + 層正規化
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # 位置編碼器
        self.pos_encoder = PositionalEncoder(d_model, max_seq_length)
        
        # Dropout層
        self.dropout = nn.Dropout(dropout)
        
        # Transformer編碼器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4,  # 調整前饋網絡維度
            dropout=dropout,
            activation='gelu',  # 使用GELU激活函數
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)  # 添加最終的層正規化
        )
        
        # 改進的注意力聚合機制
        self.attention_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # 輸出層
        if output_dim is not None:
            self.output_proj = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )
        else:
            self.output_proj = None
            
        # 初始化權重
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
        # 投影到d_model維度
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = x * math.sqrt(self.d_model)  # 縮放因子

        # 添加位置編碼
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # 通過transformer
        output = self.transformer(x, src_mask)  # (batch_size, seq_len, d_model)
        return output

    def forward(self, x, src_mask=None):
        """
        前向傳播
        Args:
            x: 輸入張量 (batch_size, seq_len, feature_dim)
            src_mask: 源序列遮罩
        Returns:
            輸出張量 (batch_size, output_dim)
        """
        output = self._encode(x, src_mask)

        # 改進的注意力加權聚合
        attention_scores = self.attention_proj(output)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # 加權求和
        weighted_output = (attention_weights * output).sum(dim=1)  # (batch_size, d_model)

        # 輸出投影
        if self.output_proj is not None:
            final_output = self.output_proj(weighted_output)
        else:
            final_output = weighted_output

        return final_output

    def get_attention_weights(self, x, src_mask=None):
        """獲取注意力權重用於可視化"""
        output = self._encode(x, src_mask)

        # 計算注意力權重
        attention_scores = self.attention_proj(output)
        attention_weights = torch.softmax(attention_scores, dim=1)

        return attention_weights
