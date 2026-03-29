# models.py - CNN, RNN, and Vision Transformer models
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============ CNN Model ============
class CNN(nn.Module):
    """1D CNN for time-series classification"""
    
    def __init__(self, input_channels=18, num_classes=12, window_size=200):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self.flat_size = 256 * (window_size // 8)
        
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input shape: (batch, window_size, channels)
        x = x.permute(0, 2, 1)  # -> (batch, channels, window_size)
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """Extract features before classification layer"""
        x = x.permute(0, 2, 1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return x


# ============ RNN (LSTM) Model ============
class RNN(nn.Module):
    """Bidirectional LSTM for time-series classification"""
    
    def __init__(self, input_channels=18, num_classes=12, hidden_size=128, num_layers=2):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch, window_size, channels)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention mechanism
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        output = self.fc(context)
        return output
    
    def get_features(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context


# ============ Vision Transformer Model ============
class PatchEmbedding(nn.Module):
    """Convert time-series to patch embeddings"""
    
    def __init__(self, input_channels=18, patch_size=20, embed_dim=128, seq_length=200):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = seq_length // patch_size
        self.projection = nn.Linear(patch_size * input_channels, embed_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        batch_size, seq_len, channels = x.shape
        
        # Reshape to patches
        x = x.reshape(batch_size, self.num_patches, self.patch_size * channels)
        x = self.projection(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder block"""
    
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer adapted for time-series"""
    
    def __init__(self, input_channels=18, num_classes=12, seq_length=200,
                 patch_size=20, embed_dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(input_channels, patch_size, embed_dim, seq_length)
        num_patches = seq_length // patch_size
        
        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder blocks
        self.encoder = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer encoder
        for block in self.encoder:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification head (use CLS token)
        cls_output = x[:, 0]
        return self.head(cls_output)
    
    def get_features(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.encoder:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


# ============ Model Factory ============
def get_model(model_name: str, num_classes: int = 12, device: str = 'cpu') -> nn.Module:
    """Factory function to get model by name"""
    models = {
        'cnn': CNN(num_classes=num_classes),
        'rnn': RNN(num_classes=num_classes),
        'vit': VisionTransformer(num_classes=num_classes)
    }
    
    if model_name.lower() not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    model = models[model_name.lower()]
    return model.to(device)
