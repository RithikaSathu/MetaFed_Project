import { useEffect } from "react";
import { useNavigate } from "react-router-dom";

export default function BackendCode(){
  const navigate = useNavigate();
  useEffect(()=> { navigate('/', { replace: true }); }, [navigate]);
  return null;
}

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Copy, Check, Download } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const CodeBlock = ({ code, filename }: { code: string; filename: string }) => {
  const [copied, setCopied] = useState(false);
  const { toast } = useToast();

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    toast({ title: "Copied!", description: `${filename} copied to clipboard` });
    setTimeout(() => setCopied(false), 2000);
  };

  const downloadFile = () => {
    const blob = new Blob([code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast({ title: "Downloaded!", description: `${filename} downloaded` });
  };

  return (
    <div className="relative">
      <div className="absolute right-2 top-2 flex gap-2 z-10">
        <Button size="sm" variant="outline" onClick={copyToClipboard}>
          {copied ? <Check className="h-4 w-4" /> : <Copy className="h-4 w-4" />}
        </Button>
        <Button size="sm" variant="outline" onClick={downloadFile}>
          <Download className="h-4 w-4" />
        </Button>
      </div>
      <ScrollArea className="h-[500px] rounded-md border bg-muted/50 p-4">
        <pre className="text-xs font-mono whitespace-pre-wrap">{code}</pre>
      </ScrollArea>
    </div>
  );
};

// ============ PYTHON CODE STRINGS ============

const requirementsTxt = `# MetaFed Backend Requirements
flask==2.3.3
flask-cors==4.0.0
numpy==1.24.3
pandas==2.0.3
torch==2.0.1
torchvision==0.15.2
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.65.0
scipy==1.11.1
`;

const configPy = `# config.py - Configuration settings
import os
import torch

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Create directories
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # PAMAP2 Dataset settings
    PAMAP2_SAMPLING_RATE = 100  # Hz
    WINDOW_SIZE = 200  # samples (2 seconds)
    STRIDE = 100  # 50% overlap
    
    # Activity labels for PAMAP2
    ACTIVITY_LABELS = {
        1: 'lying', 2: 'sitting', 3: 'standing', 4: 'walking',
        5: 'running', 6: 'cycling', 7: 'Nordic_walking',
        12: 'ascending_stairs', 13: 'descending_stairs',
        16: 'vacuum_cleaning', 17: 'ironing', 24: 'rope_jumping'
    }
    NUM_CLASSES = 12
    
    # Federation settings
    NUM_FEDERATIONS = 3
    SUBJECTS_PER_FEDERATION = 3
    
    # Training settings
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    LOCAL_EPOCHS = 5
    GLOBAL_ROUNDS = 50
    
    # FedProx settings
    FEDPROX_MU = 0.01
    
    # MetaFed settings
    META_LR = 0.01
    INNER_STEPS = 5

config = Config()
`;

const preprocessingPy = `# preprocessing.py - PAMAP2 Dataset Preprocessing
import os
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler
from config import config
import pickle

class PAMAP2Preprocessor:
    """Preprocessor for PAMAP2 dataset"""
    
    # Column names for PAMAP2 .dat files
    COLUMNS = ['timestamp', 'activity_id', 'heart_rate'] + \\
              [f'hand_temp', f'hand_acc_x', f'hand_acc_y', f'hand_acc_z',
               f'hand_gyro_x', f'hand_gyro_y', f'hand_gyro_z',
               f'hand_mag_x', f'hand_mag_y', f'hand_mag_z'] + \\
              [f'chest_temp', f'chest_acc_x', f'chest_acc_y', f'chest_acc_z',
               f'chest_gyro_x', f'chest_gyro_y', f'chest_gyro_z',
               f'chest_mag_x', f'chest_mag_y', f'chest_mag_z'] + \\
              [f'ankle_temp', f'ankle_acc_x', f'ankle_acc_y', f'ankle_acc_z',
               f'ankle_gyro_x', f'ankle_gyro_y', f'ankle_gyro_z',
               f'ankle_mag_x', f'ankle_mag_y', f'ankle_mag_z']
    
    # IMU sensor columns (excluding temperature)
    IMU_COLUMNS = [
        'hand_acc_x', 'hand_acc_y', 'hand_acc_z',
        'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
        'chest_acc_x', 'chest_acc_y', 'chest_acc_z',
        'chest_gyro_x', 'chest_gyro_y', 'chest_gyro_z',
        'ankle_acc_x', 'ankle_acc_y', 'ankle_acc_z',
        'ankle_gyro_x', 'ankle_gyro_y', 'ankle_gyro_z'
    ]
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_mapping = {}
        
    def load_subject_data(self, subject_id: int) -> pd.DataFrame:
        """Load data for a single subject"""
        filepath = os.path.join(config.RAW_DATA_DIR, f'subject10{subject_id}.dat')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        # Load .dat file (space-separated)
        df = pd.read_csv(filepath, sep=' ', header=None)
        
        # Assign column names (PAMAP2 has 54 columns)
        if df.shape[1] >= 40:
            df.columns = self.COLUMNS[:df.shape[1]]
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter data"""
        # Remove transient activities (activity_id = 0)
        df = df[df['activity_id'] != 0].copy()
        
        # Keep only activities in our label set
        valid_activities = list(config.ACTIVITY_LABELS.keys())
        df = df[df['activity_id'].isin(valid_activities)]
        
        # Handle missing values with interpolation
        df[self.IMU_COLUMNS] = df[self.IMU_COLUMNS].interpolate(method='linear', limit_direction='both')
        
        # Fill any remaining NaNs with 0
        df = df.fillna(0)
        
        return df
    
    def apply_lowpass_filter(self, data: np.ndarray, cutoff: float = 20, fs: int = 100) -> np.ndarray:
        """Apply low-pass Butterworth filter"""
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, data, axis=0)
    
    def create_windows(self, data: np.ndarray, labels: np.ndarray) -> tuple:
        """Create sliding windows from continuous data"""
        windows = []
        window_labels = []
        
        for start in range(0, len(data) - config.WINDOW_SIZE, config.STRIDE):
            end = start + config.WINDOW_SIZE
            window = data[start:end]
            
            # Use majority label for the window
            window_label = np.bincount(labels[start:end].astype(int)).argmax()
            
            windows.append(window)
            window_labels.append(window_label)
        
        return np.array(windows), np.array(window_labels)
    
    def preprocess_subject(self, subject_id: int) -> tuple:
        """Preprocess data for a single subject"""
        print(f"Processing subject {subject_id}...")
        
        # Load and clean data
        df = self.load_subject_data(subject_id)
        df = self.clean_data(df)
        
        # Extract features and labels
        features = df[self.IMU_COLUMNS].values
        labels = df['activity_id'].values
        
        # Apply low-pass filter
        features = self.apply_lowpass_filter(features)
        
        # Create sliding windows
        X, y = self.create_windows(features, labels)
        
        # Map labels to 0-11 range
        unique_labels = sorted(np.unique(y))
        self.label_mapping = {old: new for new, old in enumerate(unique_labels)}
        y = np.array([self.label_mapping.get(label, 0) for label in y])
        
        print(f"  Subject {subject_id}: {len(X)} windows, {len(np.unique(y))} classes")
        
        return X, y
    
    def preprocess_all_subjects(self, subjects: list = None) -> dict:
        """Preprocess all subjects and organize by federation"""
        if subjects is None:
            subjects = list(range(1, 10))  # Subjects 1-9
        
        all_data = {}
        
        for subject_id in subjects:
            try:
                X, y = self.preprocess_subject(subject_id)
                all_data[subject_id] = {'X': X, 'y': y}
            except FileNotFoundError as e:
                print(f"  Skipping subject {subject_id}: {e}")
                continue
        
        return all_data
    
    def create_federations(self, all_data: dict) -> dict:
        """Organize data into 3 federations"""
        subject_ids = list(all_data.keys())
        
        # Distribute subjects across federations
        federations = {
            'federation_1': subject_ids[:3],   # First 3 subjects
            'federation_2': subject_ids[3:6],  # Next 3 subjects
            'federation_3': subject_ids[6:9]   # Last 3 subjects
        }
        
        federation_data = {}
        
        for fed_name, fed_subjects in federations.items():
            X_fed = []
            y_fed = []
            
            for subj_id in fed_subjects:
                if subj_id in all_data:
                    X_fed.append(all_data[subj_id]['X'])
                    y_fed.append(all_data[subj_id]['y'])
            
            if X_fed:
                X_fed = np.concatenate(X_fed, axis=0)
                y_fed = np.concatenate(y_fed, axis=0)
                
                # Normalize features
                original_shape = X_fed.shape
                X_fed_flat = X_fed.reshape(-1, X_fed.shape[-1])
                X_fed_normalized = self.scaler.fit_transform(X_fed_flat)
                X_fed = X_fed_normalized.reshape(original_shape)
                
                federation_data[fed_name] = {
                    'X': X_fed.astype(np.float32),
                    'y': y_fed.astype(np.int64),
                    'subjects': fed_subjects
                }
                
                print(f"{fed_name}: {len(X_fed)} samples from subjects {fed_subjects}")
        
        return federation_data
    
    def save_processed_data(self, federation_data: dict):
        """Save processed data to disk"""
        output_path = os.path.join(config.PROCESSED_DATA_DIR, 'federation_data.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(federation_data, f)
        print(f"Saved processed data to {output_path}")
    
    def load_processed_data(self) -> dict:
        """Load processed data from disk"""
        input_path = os.path.join(config.PROCESSED_DATA_DIR, 'federation_data.pkl')
        with open(input_path, 'rb') as f:
            return pickle.load(f)

def main():
    """Main preprocessing pipeline"""
    preprocessor = PAMAP2Preprocessor()
    
    print("="*50)
    print("PAMAP2 Dataset Preprocessing")
    print("="*50)
    
    # Preprocess all subjects
    all_data = preprocessor.preprocess_all_subjects()
    
    if not all_data:
        print("\\nNo data found! Please ensure PAMAP2 .dat files are in:")
        print(f"  {config.RAW_DATA_DIR}")
        print("\\nExpected files: subject101.dat, subject102.dat, ..., subject109.dat")
        return
    
    # Create federations
    print("\\nCreating federations...")
    federation_data = preprocessor.create_federations(all_data)
    
    # Save processed data
    preprocessor.save_processed_data(federation_data)
    
    print("\\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)

if __name__ == '__main__':
    main()
`;

const modelsPy = `# models.py - CNN, RNN, and Vision Transformer models
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
`;

const algorithmsPy = `# algorithms.py - Federated Learning Algorithms
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset
from config import config
from tqdm import tqdm

# ============ FedAvg Algorithm ============
class FedAvg:
    """Federated Averaging Algorithm"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.device = device
        self.global_model = model.to(device)
        
    def train_local(self, model: nn.Module, train_loader: DataLoader, 
                    epochs: int = 5, lr: float = 0.001) -> Tuple[nn.Module, float]:
        """Train model locally on client data"""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            total_loss += epoch_loss / len(train_loader)
        
        return model, total_loss / epochs
    
    def aggregate(self, local_models: List[nn.Module], weights: List[float] = None):
        """Aggregate local models using weighted averaging"""
        if weights is None:
            weights = [1.0 / len(local_models)] * len(local_models)
        
        # Get global model state dict
        global_state = self.global_model.state_dict()
        
        # Initialize aggregated state
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        
        # Weighted average of local models
        for model, weight in zip(local_models, weights):
            local_state = model.state_dict()
            for key in global_state.keys():
                global_state[key] += weight * local_state[key].float()
        
        self.global_model.load_state_dict(global_state)
        return self.global_model
    
    def get_global_model(self) -> nn.Module:
        return copy.deepcopy(self.global_model)


# ============ FedBN Algorithm ============
class FedBN(FedAvg):
    """FedBN: Federated Learning with Local Batch Normalization"""
    
    def aggregate(self, local_models: List[nn.Module], weights: List[float] = None):
        """Aggregate only non-BN layers"""
        if weights is None:
            weights = [1.0 / len(local_models)] * len(local_models)
        
        global_state = self.global_model.state_dict()
        
        # Identify BatchNorm parameters
        bn_keys = [key for key in global_state.keys() 
                   if 'bn' in key.lower() or 'norm' in key.lower()]
        
        # Initialize non-BN parameters
        for key in global_state.keys():
            if key not in bn_keys:
                global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        
        # Average only non-BN parameters
        for model, weight in zip(local_models, weights):
            local_state = model.state_dict()
            for key in global_state.keys():
                if key not in bn_keys:
                    global_state[key] += weight * local_state[key].float()
        
        # Load aggregated state (BN params remain unchanged)
        current_state = self.global_model.state_dict()
        for key in bn_keys:
            global_state[key] = current_state[key]
        
        self.global_model.load_state_dict(global_state)
        return self.global_model


# ============ FedProx Algorithm ============
class FedProx(FedAvg):
    """FedProx: Federated Optimization with Proximal Term"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', mu: float = 0.01):
        super().__init__(model, device)
        self.mu = mu
    
    def train_local(self, model: nn.Module, train_loader: DataLoader,
                    epochs: int = 5, lr: float = 0.001) -> Tuple[nn.Module, float]:
        """Train with proximal term"""
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Store global model parameters for proximal term
        global_params = {name: param.clone().detach() 
                        for name, param in self.global_model.named_parameters()}
        
        total_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                # Cross-entropy loss
                ce_loss = criterion(outputs, y_batch)
                
                # Proximal term: mu/2 * ||w - w_global||^2
                prox_loss = 0
                for name, param in model.named_parameters():
                    prox_loss += ((param - global_params[name]) ** 2).sum()
                prox_loss = (self.mu / 2) * prox_loss
                
                loss = ce_loss + prox_loss
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            total_loss += epoch_loss / len(train_loader)
        
        return model, total_loss / epochs


# ============ MetaFed Algorithm (Homogeneous) ============
class MetaFed:
    """MetaFed: Meta-Learning based Federated Learning"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu', 
                 meta_lr: float = 0.01, inner_lr: float = 0.001, inner_steps: int = 5):
        self.device = device
        self.global_model = model.to(device)
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        
    def train_local(self, model: nn.Module, train_loader: DataLoader,
                    epochs: int = 5, lr: float = 0.001) -> Tuple[nn.Module, float]:
        """Inner loop: Task-specific adaptation"""
        model.train()
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Create fast weights for inner loop
                fast_weights = {name: param.clone() for name, param in model.named_parameters()}
                
                # Inner loop adaptation
                for _ in range(self.inner_steps):
                    outputs = self._forward_with_weights(model, X_batch, fast_weights)
                    loss = criterion(outputs, y_batch)
                    
                    # Compute gradients
                    grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
                    
                    # Update fast weights
                    fast_weights = {name: param - self.inner_lr * grad 
                                   for (name, param), grad in zip(fast_weights.items(), grads)}
                
                # Update model with fast weights
                for name, param in model.named_parameters():
                    param.data = fast_weights[name].data
                
                total_loss += loss.item()
        
        return model, total_loss / (epochs * len(train_loader))
    
    def _forward_with_weights(self, model: nn.Module, x: torch.Tensor, 
                              weights: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass using custom weights"""
        # Store original weights
        original_weights = {name: param.clone() for name, param in model.named_parameters()}
        
        # Load custom weights
        for name, param in model.named_parameters():
            param.data = weights[name]
        
        # Forward pass
        output = model(x)
        
        # Restore original weights
        for name, param in model.named_parameters():
            param.data = original_weights[name]
        
        return output
    
    def aggregate(self, local_models: List[nn.Module], weights: List[float] = None):
        """Meta-aggregation with adaptive weighting"""
        if weights is None:
            weights = [1.0 / len(local_models)] * len(local_models)
        
        global_state = self.global_model.state_dict()
        
        for key in global_state.keys():
            global_state[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
        
        for model, weight in zip(local_models, weights):
            local_state = model.state_dict()
            for key in global_state.keys():
                global_state[key] += weight * local_state[key].float()
        
        self.global_model.load_state_dict(global_state)
        return self.global_model
    
    def get_global_model(self) -> nn.Module:
        return copy.deepcopy(self.global_model)


# ============ MetaFed Heterogeneous (Extension) ============
class MetaFedHeterogeneous:
    """MetaFed with heterogeneous models across federations"""
    
    def __init__(self, models: Dict[str, nn.Module], device: str = 'cpu',
                 meta_lr: float = 0.01, inner_lr: float = 0.001):
        self.device = device
        self.models = {name: model.to(device) for name, model in models.items()}
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        
        # Knowledge distillation components
        self.temperature = 3.0
        self.alpha = 0.5  # Weight for distillation loss
        
    def train_local(self, model_name: str, train_loader: DataLoader,
                    epochs: int = 5) -> Tuple[nn.Module, float]:
        """Train specific model with knowledge distillation"""
        model = self.models[model_name]
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=self.inner_lr)
        criterion = nn.CrossEntropyLoss()
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        
        # Get ensemble predictions for distillation
        ensemble_logits = self._get_ensemble_predictions(train_loader)
        
        total_loss = 0
        batch_idx = 0
        
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                
                # Task loss
                task_loss = criterion(outputs, y_batch)
                
                # Knowledge distillation loss
                if ensemble_logits is not None and batch_idx < len(ensemble_logits):
                    soft_targets = F.softmax(ensemble_logits[batch_idx] / self.temperature, dim=1)
                    soft_outputs = F.log_softmax(outputs / self.temperature, dim=1)
                    distill_loss = kl_criterion(soft_outputs, soft_targets) * (self.temperature ** 2)
                    
                    loss = (1 - self.alpha) * task_loss + self.alpha * distill_loss
                else:
                    loss = task_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_idx += 1
        
        return model, total_loss / (epochs * len(train_loader))
    
    def _get_ensemble_predictions(self, data_loader: DataLoader) -> List[torch.Tensor]:
        """Get ensemble predictions from all models"""
        ensemble_logits = []
        
        for X_batch, _ in data_loader:
            X_batch = X_batch.to(self.device)
            
            batch_logits = []
            for model in self.models.values():
                model.eval()
                with torch.no_grad():
                    logits = model(X_batch)
                    batch_logits.append(logits)
            
            # Average logits
            avg_logits = torch.stack(batch_logits).mean(dim=0)
            ensemble_logits.append(avg_logits)
        
        return ensemble_logits
    
    def aggregate_knowledge(self):
        """Aggregate knowledge across heterogeneous models using distillation"""
        # Each model learns from ensemble, no parameter averaging needed
        pass
    
    def get_models(self) -> Dict[str, nn.Module]:
        return {name: copy.deepcopy(model) for name, model in self.models.items()}


# Import F for heterogeneous metafed
import torch.nn.functional as F
`;

const trainerPy = `# trainer.py - Training orchestration
import torch
import torch.nn as nn
import numpy as np
import copy
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from config import config
from models import get_model, CNN, RNN, VisionTransformer
from algorithms import FedAvg, FedBN, FedProx, MetaFed, MetaFedHeterogeneous


class FederatedTrainer:
    """Orchestrates federated learning training"""
    
    def __init__(self, federation_data: Dict, algorithm: str = 'fedavg', 
                 model_name: str = 'cnn', device: str = None):
        self.device = device or str(config.DEVICE)
        self.federation_data = federation_data
        self.algorithm_name = algorithm.lower()
        self.model_name = model_name.lower()
        
        # Create data loaders
        self.train_loaders, self.test_loaders = self._create_data_loaders()
        
        # Initialize model and algorithm
        self.model = get_model(model_name, config.NUM_CLASSES, self.device)
        self.algorithm = self._init_algorithm()
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_accuracy': [],
            'test_precision': [],
            'test_recall': [],
            'test_f1': [],
            'rounds': []
        }
        
    def _create_data_loaders(self) -> Tuple[Dict, Dict]:
        """Create train/test data loaders for each federation"""
        train_loaders = {}
        test_loaders = {}
        
        for fed_name, data in self.federation_data.items():
            X = torch.tensor(data['X'], dtype=torch.float32)
            y = torch.tensor(data['y'], dtype=torch.long)
            
            dataset = TensorDataset(X, y)
            
            # 80-20 train-test split
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            
            train_loaders[fed_name] = DataLoader(
                train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
            )
            test_loaders[fed_name] = DataLoader(
                test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
            )
        
        return train_loaders, test_loaders
    
    def _init_algorithm(self):
        """Initialize the federated learning algorithm"""
        if self.algorithm_name == 'fedavg':
            return FedAvg(copy.deepcopy(self.model), self.device)
        elif self.algorithm_name == 'fedbn':
            return FedBN(copy.deepcopy(self.model), self.device)
        elif self.algorithm_name == 'fedprox':
            return FedProx(copy.deepcopy(self.model), self.device, mu=config.FEDPROX_MU)
        elif self.algorithm_name == 'metafed':
            return MetaFed(copy.deepcopy(self.model), self.device, 
                          meta_lr=config.META_LR, inner_lr=config.LEARNING_RATE)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm_name}")
    
    def train(self, num_rounds: int = None) -> Dict:
        """Run federated training"""
        num_rounds = num_rounds or config.GLOBAL_ROUNDS
        
        print(f"\\n{'='*60}")
        print(f"Starting Federated Learning Training")
        print(f"Algorithm: {self.algorithm_name.upper()}")
        print(f"Model: {self.model_name.upper()}")
        print(f"Device: {self.device}")
        print(f"Rounds: {num_rounds}")
        print(f"{'='*60}\\n")
        
        for round_idx in tqdm(range(num_rounds), desc="Training Rounds"):
            # Local training on each federation
            local_models = []
            local_losses = []
            weights = []
            
            for fed_name, train_loader in self.train_loaders.items():
                # Get local model copy
                local_model = self.algorithm.get_global_model()
                
                # Train locally
                trained_model, loss = self.algorithm.train_local(
                    local_model, train_loader,
                    epochs=config.LOCAL_EPOCHS,
                    lr=config.LEARNING_RATE
                )
                
                local_models.append(trained_model)
                local_losses.append(loss)
                weights.append(len(train_loader.dataset))
            
            # Normalize weights
            total_samples = sum(weights)
            weights = [w / total_samples for w in weights]
            
            # Aggregate models
            self.algorithm.aggregate(local_models, weights)
            
            # Evaluate
            avg_loss = np.mean(local_losses)
            metrics = self.evaluate()
            
            # Record history
            self.history['train_loss'].append(avg_loss)
            self.history['test_accuracy'].append(metrics['accuracy'])
            self.history['test_precision'].append(metrics['precision'])
            self.history['test_recall'].append(metrics['recall'])
            self.history['test_f1'].append(metrics['f1'])
            self.history['rounds'].append(round_idx + 1)
            
            if (round_idx + 1) % 10 == 0:
                print(f"Round {round_idx + 1}: Loss={avg_loss:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
        
        print(f"\\nTraining Complete!")
        return self.history
    
    def evaluate(self) -> Dict:
        """Evaluate global model on all test data"""
        model = self.algorithm.get_global_model()
        model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for fed_name, test_loader in self.test_loaders.items():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    outputs = model(X_batch)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    all_preds.extend(preds)
                    all_labels.extend(y_batch.numpy())
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist()
        }
        
        return metrics
    
    def save_results(self, filepath: str = None):
        """Save training results"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(
                config.LOGS_DIR, 
                f"{self.algorithm_name}_{self.model_name}_{timestamp}.json"
            )
        
        results = {
            'algorithm': self.algorithm_name,
            'model': self.model_name,
            'config': {
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE,
                'local_epochs': config.LOCAL_EPOCHS,
                'global_rounds': config.GLOBAL_ROUNDS
            },
            'history': self.history,
            'final_metrics': self.evaluate()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filepath}")
        return filepath


class HeterogeneousFederatedTrainer:
    """Trainer for heterogeneous MetaFed (different models per federation)"""
    
    def __init__(self, federation_data: Dict, device: str = None):
        self.device = device or str(config.DEVICE)
        self.federation_data = federation_data
        
        # Assign different models to federations
        self.model_assignment = {
            'federation_1': 'cnn',
            'federation_2': 'rnn',
            'federation_3': 'vit'
        }
        
        # Create data loaders
        self.train_loaders, self.test_loaders = self._create_data_loaders()
        
        # Initialize models
        self.models = {
            'cnn': get_model('cnn', config.NUM_CLASSES, self.device),
            'rnn': get_model('rnn', config.NUM_CLASSES, self.device),
            'vit': get_model('vit', config.NUM_CLASSES, self.device)
        }
        
        # Initialize MetaFed Heterogeneous
        self.algorithm = MetaFedHeterogeneous(self.models, self.device)
        
        self.history = {fed: {'train_loss': [], 'test_accuracy': [], 'test_f1': []}
                       for fed in self.federation_data.keys()}
    
    def _create_data_loaders(self):
        """Create data loaders for each federation"""
        train_loaders = {}
        test_loaders = {}
        
        for fed_name, data in self.federation_data.items():
            X = torch.tensor(data['X'], dtype=torch.float32)
            y = torch.tensor(data['y'], dtype=torch.long)
            
            dataset = TensorDataset(X, y)
            train_size = int(0.8 * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
            
            train_loaders[fed_name] = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            test_loaders[fed_name] = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        return train_loaders, test_loaders
    
    def train(self, num_rounds: int = None) -> Dict:
        """Train heterogeneous federation"""
        num_rounds = num_rounds or config.GLOBAL_ROUNDS
        
        print(f"\\n{'='*60}")
        print(f"MetaFed Heterogeneous Training")
        print(f"Model Assignment: {self.model_assignment}")
        print(f"{'='*60}\\n")
        
        for round_idx in tqdm(range(num_rounds), desc="Training Rounds"):
            for fed_name, train_loader in self.train_loaders.items():
                model_name = self.model_assignment[fed_name]
                
                # Train local model
                model, loss = self.algorithm.train_local(model_name, train_loader, epochs=config.LOCAL_EPOCHS)
                
                # Evaluate
                metrics = self._evaluate_federation(fed_name)
                
                # Record history
                self.history[fed_name]['train_loss'].append(loss)
                self.history[fed_name]['test_accuracy'].append(metrics['accuracy'])
                self.history[fed_name]['test_f1'].append(metrics['f1'])
            
            # Knowledge aggregation
            self.algorithm.aggregate_knowledge()
        
        return self.history
    
    def _evaluate_federation(self, fed_name: str) -> Dict:
        """Evaluate specific federation model"""
        model_name = self.model_assignment[fed_name]
        model = self.models[model_name]
        model.eval()
        
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loaders[fed_name]:
                X_batch = X_batch.to(self.device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        }
    
    def get_all_metrics(self) -> Dict:
        """Get final metrics for all federations"""
        results = {}
        for fed_name in self.federation_data.keys():
            results[fed_name] = {
                'model': self.model_assignment[fed_name],
                'metrics': self._evaluate_federation(fed_name)
            }
        return results
`;

const flaskAppPy = `# app.py - Flask API Backend
from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import json
import os
import threading
from datetime import datetime

from config import config
from preprocessing import PAMAP2Preprocessor
from trainer import FederatedTrainer, HeterogeneousFederatedTrainer

app = Flask(__name__)
CORS(app)

# Global state
training_status = {
    'is_training': False,
    'current_algorithm': None,
    'progress': 0,
    'message': ''
}
training_results = {}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(config.DEVICE),
        'cuda_available': torch.cuda.is_available()
    })


@app.route('/api/preprocess', methods=['POST'])
def preprocess_data():
    """Preprocess PAMAP2 dataset"""
    try:
        preprocessor = PAMAP2Preprocessor()
        all_data = preprocessor.preprocess_all_subjects()
        
        if not all_data:
            return jsonify({
                'success': False,
                'error': 'No data found. Ensure PAMAP2 .dat files are in data/raw/'
            }), 400
        
        federation_data = preprocessor.create_federations(all_data)
        preprocessor.save_processed_data(federation_data)
        
        summary = {fed: {'samples': len(data['X']), 'subjects': data['subjects']}
                   for fed, data in federation_data.items()}
        
        return jsonify({
            'success': True,
            'message': 'Preprocessing complete',
            'federations': summary
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/train', methods=['POST'])
def start_training():
    """Start federated learning training"""
    global training_status, training_results
    
    if training_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'}), 400
    
    data = request.json
    algorithm = data.get('algorithm', 'fedavg')
    model = data.get('model', 'cnn')
    rounds = data.get('rounds', config.GLOBAL_ROUNDS)
    
    def train_async():
        global training_status, training_results
        training_status = {
            'is_training': True,
            'current_algorithm': algorithm,
            'progress': 0,
            'message': 'Loading data...'
        }
        
        try:
            # Load preprocessed data
            preprocessor = PAMAP2Preprocessor()
            federation_data = preprocessor.load_processed_data()
            
            training_status['message'] = 'Initializing trainer...'
            
            # Create trainer
            trainer = FederatedTrainer(
                federation_data, 
                algorithm=algorithm,
                model_name=model,
                device=str(config.DEVICE)
            )
            
            # Train
            training_status['message'] = 'Training in progress...'
            history = trainer.train(num_rounds=rounds)
            
            # Get final metrics
            final_metrics = trainer.evaluate()
            
            # Save results
            result_key = f"{algorithm}_{model}"
            training_results[result_key] = {
                'algorithm': algorithm,
                'model': model,
                'history': history,
                'final_metrics': final_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            trainer.save_results()
            
            training_status['progress'] = 100
            training_status['message'] = 'Training complete!'
            
        except Exception as e:
            training_status['message'] = f'Error: {str(e)}'
        finally:
            training_status['is_training'] = False
    
    thread = threading.Thread(target=train_async)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': f'Training started: {algorithm} with {model}'
    })


@app.route('/api/train/heterogeneous', methods=['POST'])
def start_heterogeneous_training():
    """Start heterogeneous MetaFed training"""
    global training_status, training_results
    
    if training_status['is_training']:
        return jsonify({'success': False, 'error': 'Training already in progress'}), 400
    
    data = request.json
    rounds = data.get('rounds', config.GLOBAL_ROUNDS)
    
    def train_async():
        global training_status, training_results
        training_status = {
            'is_training': True,
            'current_algorithm': 'metafed_heterogeneous',
            'progress': 0,
            'message': 'Loading data...'
        }
        
        try:
            preprocessor = PAMAP2Preprocessor()
            federation_data = preprocessor.load_processed_data()
            
            trainer = HeterogeneousFederatedTrainer(federation_data, device=str(config.DEVICE))
            
            training_status['message'] = 'Training heterogeneous models...'
            history = trainer.train(num_rounds=rounds)
            
            final_metrics = trainer.get_all_metrics()
            
            training_results['metafed_heterogeneous'] = {
                'algorithm': 'metafed_heterogeneous',
                'models': trainer.model_assignment,
                'history': history,
                'final_metrics': final_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            training_status['progress'] = 100
            training_status['message'] = 'Training complete!'
            
        except Exception as e:
            training_status['message'] = f'Error: {str(e)}'
        finally:
            training_status['is_training'] = False
    
    thread = threading.Thread(target=train_async)
    thread.start()
    
    return jsonify({
        'success': True,
        'message': 'Heterogeneous MetaFed training started'
    })


@app.route('/api/train/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    return jsonify(training_status)


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get all training results"""
    return jsonify({
        'success': True,
        'results': training_results
    })


@app.route('/api/results/<algorithm>', methods=['GET'])
def get_algorithm_results(algorithm):
    """Get results for specific algorithm"""
    if algorithm in training_results:
        return jsonify({
            'success': True,
            'result': training_results[algorithm]
        })
    return jsonify({'success': False, 'error': 'Results not found'}), 404


@app.route('/api/compare', methods=['POST'])
def compare_algorithms():
    """Compare multiple algorithms"""
    data = request.json
    algorithms = data.get('algorithms', [])
    
    comparison = {}
    for algo in algorithms:
        if algo in training_results:
            comparison[algo] = {
                'accuracy': training_results[algo]['final_metrics'].get('accuracy'),
                'f1': training_results[algo]['final_metrics'].get('f1'),
                'precision': training_results[algo]['final_metrics'].get('precision'),
                'recall': training_results[algo]['final_metrics'].get('recall')
            }
    
    return jsonify({
        'success': True,
        'comparison': comparison
    })


@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get available log files"""
    log_files = []
    if os.path.exists(config.LOGS_DIR):
        log_files = [f for f in os.listdir(config.LOGS_DIR) if f.endswith('.json')]
    return jsonify({'success': True, 'logs': log_files})


if __name__ == '__main__':
    print(f"Starting MetaFed Backend Server")
    print(f"Device: {config.DEVICE}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    app.run(host='0.0.0.0', port=5000, debug=True)
`;

const runAllPy = `# run_all_experiments.py - Run all FL algorithms and compare
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from config import config
from preprocessing import PAMAP2Preprocessor
from trainer import FederatedTrainer, HeterogeneousFederatedTrainer


def run_all_experiments():
    """Run all federated learning experiments"""
    
    # Load data
    print("Loading preprocessed data...")
    preprocessor = PAMAP2Preprocessor()
    
    try:
        federation_data = preprocessor.load_processed_data()
    except FileNotFoundError:
        print("Preprocessed data not found. Running preprocessing...")
        all_data = preprocessor.preprocess_all_subjects()
        federation_data = preprocessor.create_federations(all_data)
        preprocessor.save_processed_data(federation_data)
    
    # Algorithms to test
    algorithms = ['fedavg', 'fedbn', 'fedprox', 'metafed']
    model = 'cnn'
    num_rounds = 50
    
    all_results = {}
    
    # Run homogeneous experiments
    print("\\n" + "="*60)
    print("RUNNING HOMOGENEOUS FL EXPERIMENTS")
    print("="*60)
    
    for algo in algorithms:
        print(f"\\n>>> Training {algo.upper()} with {model.upper()}")
        
        trainer = FederatedTrainer(
            federation_data,
            algorithm=algo,
            model_name=model,
            device=str(config.DEVICE)
        )
        
        history = trainer.train(num_rounds=num_rounds)
        final_metrics = trainer.evaluate()
        
        all_results[algo] = {
            'history': history,
            'final_metrics': final_metrics
        }
        
        trainer.save_results()
    
    # Run heterogeneous experiment
    print("\\n" + "="*60)
    print("RUNNING HETEROGENEOUS METAFED EXPERIMENT")
    print("="*60)
    
    hetero_trainer = HeterogeneousFederatedTrainer(federation_data, device=str(config.DEVICE))
    hetero_history = hetero_trainer.train(num_rounds=num_rounds)
    hetero_metrics = hetero_trainer.get_all_metrics()
    
    all_results['metafed_heterogeneous'] = {
        'history': hetero_history,
        'final_metrics': hetero_metrics
    }
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(config.LOGS_DIR, f"comparison_{timestamp}.json")
    
    # Convert numpy arrays for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(comparison_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)
    
    print(f"\\nAll results saved to {comparison_file}")
    
    # Generate comparison plots
    generate_comparison_plots(all_results)
    
    return all_results


def generate_comparison_plots(results: dict):
    """Generate comparison visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    algorithms = ['fedavg', 'fedbn', 'fedprox', 'metafed']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # Plot 1: Training Loss
    ax1 = axes[0, 0]
    for algo, color in zip(algorithms, colors):
        if algo in results:
            losses = results[algo]['history']['train_loss']
            ax1.plot(losses, label=algo.upper(), color=color, linewidth=2)
    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Test Accuracy
    ax2 = axes[0, 1]
    for algo, color in zip(algorithms, colors):
        if algo in results:
            acc = results[algo]['history']['test_accuracy']
            ax2.plot(acc, label=algo.upper(), color=color, linewidth=2)
    ax2.set_xlabel('Communication Round')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Metrics Bar Chart
    ax3 = axes[1, 0]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (algo, color) in enumerate(zip(algorithms, colors)):
        if algo in results:
            values = [results[algo]['final_metrics'].get(m, 0) for m in metrics]
            ax3.bar(x + i*width, values, width, label=algo.upper(), color=color)
    
    ax3.set_xlabel('Metric')
    ax3.set_ylabel('Score')
    ax3.set_title('Final Metrics Comparison')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels([m.capitalize() for m in metrics])
    ax3.legend()
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: F1 Score Over Time
    ax4 = axes[1, 1]
    for algo, color in zip(algorithms, colors):
        if algo in results:
            f1 = results[algo]['history']['test_f1']
            ax4.plot(f1, label=algo.upper(), color=color, linewidth=2)
    ax4.set_xlabel('Communication Round')
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(config.LOGS_DIR, 'comparison_plots.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    print(f"Comparison plots saved to {plot_path}")


if __name__ == '__main__':
    run_all_experiments()
`;

const setupInstructions = `# MetaFed Backend Setup Instructions

## Quick Start

### 1. Create Project Structure
\`\`\`bash
mkdir MetaFed-Heterogeneous-FL
cd MetaFed-Heterogeneous-FL
mkdir -p backend/{data/{raw,processed},saved_models,logs}
\`\`\`

### 2. Copy Python Files
Save all the Python files from this page to the \`backend/\` folder:
- config.py
- preprocessing.py
- models.py
- algorithms.py
- trainer.py
- app.py
- run_all_experiments.py
- requirements.txt

### 3. Setup Environment
\`\`\`bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 4. Download PAMAP2 Dataset
1. Go to: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
2. Download the dataset
3. Extract \`subject101.dat\` through \`subject109.dat\` to \`backend/data/raw/\`

### 5. Run Preprocessing
\`\`\`bash
python preprocessing.py
\`\`\`

### 6. Run All Experiments
\`\`\`bash
python run_all_experiments.py
\`\`\`
This runs FedAvg, FedBN, FedProx, MetaFed, and MetaFed-Heterogeneous.

### 7. Start Flask API (for React frontend)
\`\`\`bash
python app.py
\`\`\`
API will be available at http://localhost:5000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/health | GET | Health check |
| /api/preprocess | POST | Preprocess PAMAP2 data |
| /api/train | POST | Start training (algorithm, model, rounds) |
| /api/train/heterogeneous | POST | Start heterogeneous MetaFed |
| /api/train/status | GET | Get training status |
| /api/results | GET | Get all results |
| /api/results/<algo> | GET | Get specific algorithm results |
| /api/compare | POST | Compare algorithms |

## GPU Usage

The code automatically detects CUDA. To force CPU:
\`\`\`python
# In config.py
DEVICE = torch.device('cpu')
\`\`\`

## Troubleshooting

1. **Out of Memory**: Reduce BATCH_SIZE in config.py
2. **No CUDA**: Install CUDA toolkit and pytorch with CUDA support
3. **Missing Data**: Ensure .dat files are in data/raw/
`;

export default function BackendCode() {
  return (
    <div className="min-h-screen bg-background p-6">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-foreground mb-2">
          MetaFed Backend - Python Code
        </h1>
        <p className="text-muted-foreground mb-6">
          Complete Python backend for Federated Learning with PAMAP2 dataset
        </p>
        
        <Tabs defaultValue="setup" className="w-full">
          <TabsList className="grid grid-cols-4 lg:grid-cols-8 mb-4">
            <TabsTrigger value="setup">Setup</TabsTrigger>
            <TabsTrigger value="requirements">Requirements</TabsTrigger>
            <TabsTrigger value="config">Config</TabsTrigger>
            <TabsTrigger value="preprocessing">Preprocessing</TabsTrigger>
            <TabsTrigger value="models">Models</TabsTrigger>
            <TabsTrigger value="algorithms">Algorithms</TabsTrigger>
            <TabsTrigger value="trainer">Trainer</TabsTrigger>
            <TabsTrigger value="app">Flask API</TabsTrigger>
          </TabsList>
          
          <TabsContent value="setup">
            <Card>
              <CardHeader>
                <CardTitle>Setup Instructions</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={setupInstructions} filename="SETUP.md" />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="requirements">
            <Card>
              <CardHeader>
                <CardTitle>requirements.txt</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={requirementsTxt} filename="requirements.txt" />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="config">
            <Card>
              <CardHeader>
                <CardTitle>config.py</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={configPy} filename="config.py" />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="preprocessing">
            <Card>
              <CardHeader>
                <CardTitle>preprocessing.py</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={preprocessingPy} filename="preprocessing.py" />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="models">
            <Card>
              <CardHeader>
                <CardTitle>models.py - CNN, RNN, Vision Transformer</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={modelsPy} filename="models.py" />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="algorithms">
            <Card>
              <CardHeader>
                <CardTitle>algorithms.py - FedAvg, FedBN, FedProx, MetaFed</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={algorithmsPy} filename="algorithms.py" />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="trainer">
            <Card>
              <CardHeader>
                <CardTitle>trainer.py + run_all_experiments.py</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={trainerPy + "\n\n" + runAllPy} filename="trainer.py" />
              </CardContent>
            </Card>
          </TabsContent>
          
          <TabsContent value="app">
            <Card>
              <CardHeader>
                <CardTitle>app.py - Flask API</CardTitle>
              </CardHeader>
              <CardContent>
                <CodeBlock code={flaskAppPy} filename="app.py" />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}
