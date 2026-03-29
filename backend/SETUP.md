# MetaFed Backend Setup Instructions

## Quick Start

### 1. Create Project Structure
```bash
mkdir MetaFed-Heterogeneous-FL
cd MetaFed-Heterogeneous-FL
mkdir -p backend/{data/{raw,processed},saved_models,logs}
```

### 2. Copy Python Files
Save all the Python files from this page to the `backend/` folder:
- config.py
- preprocessing.py
- models.py
- algorithms.py
- trainer.py
- app.py
- run_all_experiments.py
- requirements.txt

### 3. Setup Environment
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Download PAMAP2 Dataset
1. Go to: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
2. Download the dataset
3. Extract `subject101.dat` through `subject109.dat` to `backend/data/raw/`

### 5. Run Preprocessing
```bash
python preprocessing.py
```

### 6. Run All Experiments
```bash
python run_all_experiments.py
```
This runs FedAvg, FedBN, FedProx, MetaFed, and MetaFed-Heterogeneous.

### 7. Start Flask API (for React frontend)
```bash
python app.py
```
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
```python
# In config.py
DEVICE = torch.device('cpu')
```

## Troubleshooting

1. **Out of Memory**: Reduce BATCH_SIZE in config.py
2. **No CUDA**: Install CUDA toolkit and pytorch with CUDA support
3. **Missing Data**: Ensure .dat files are in data/raw/
