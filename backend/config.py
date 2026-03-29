# config.py - Configuration settings
import os

# Optional torch import: allow running the server without torch installed
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    TORCH_AVAILABLE = False

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
    
    # Device configuration (graceful fallback if torch isn't available)
    if TORCH_AVAILABLE:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # Use a simple string when torch is unavailable
        DEVICE = 'cpu'
    
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
