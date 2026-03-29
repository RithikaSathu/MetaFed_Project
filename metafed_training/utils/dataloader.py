import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_pamap2_federations(data_path, sequence_length=100, num_federations=3, max_samples_per_fed=None):
    """
    Load PAMAP2 dataset and split into federations by subject

    Expected file format: CSV with columns for sensor readings and activity labels
    Subjects are split across federations to simulate non-IID data

    Parameters:
    - max_samples_per_fed: if set, randomly sample up to this many sequences per federation (helps memory/time)
    """
    # Subject IDs for each federation (adjust based on your data)
    federation_subjects = [
        [1, 2, 3],      # Federation 1
        [4, 5, 6],      # Federation 2  
        [7, 8, 9],      # Federation 3
    ]
    
    federations = []
    
    for fed_idx, subjects in enumerate(federation_subjects[:num_federations]):
        X_fed, y_fed = [], []
        
        for subject_id in subjects:
            file_path = f"{data_path}/subject10{subject_id}.dat"
            try:
                # Load PAMAP2 format
                # Use a lighter read that handles whitespace better
                data = pd.read_csv(file_path, sep='\s+', header=None, engine='python')
                
                # Column 1: activity label, Columns 4-52+: sensor data
                labels = data.iloc[:, 1].values
                features = data.iloc[:, 4:].values
                
                # Remove NaN and transient activities (label 0)
                valid_mask = (labels > 0) & (~np.isnan(features).any(axis=1))
                labels = labels[valid_mask]
                features = features[valid_mask]
                
                # Create sequences
                for i in range(0, len(features) - sequence_length, max(1, sequence_length // 2)):
                    seq = features[i:i + sequence_length]
                    label = int(labels[i + sequence_length // 2])
                    if len(seq) == sequence_length:
                        X_fed.append(seq)
                        y_fed.append(label - 1)  # 0-indexed
                        
            except FileNotFoundError:
                print(f"Warning: {file_path} not found, skipping...")
            except Exception as e:
                print(f"Warning: error reading {file_path}: {e}")
        
        if len(X_fed) > 0:
            X_fed = np.array(X_fed, dtype=np.float32)
            y_fed = np.array(y_fed, dtype=np.int64)

            # Optionally subsample to limit memory/time
            if max_samples_per_fed is not None and len(X_fed) > max_samples_per_fed:
                idx = np.random.choice(len(X_fed), max_samples_per_fed, replace=False)
                X_fed = X_fed[idx]
                y_fed = y_fed[idx]
            
            # Normalize features
            scaler = StandardScaler()
            X_fed = scaler.fit_transform(X_fed.reshape(-1, X_fed.shape[-1])).reshape(X_fed.shape)
            
            federations.append((X_fed, y_fed))
            print(f"Federation {fed_idx + 1}: {len(X_fed)} samples, {len(np.unique(y_fed))} classes")
    
    return federations

def create_data_loaders(federations, batch_size=32, test_split=0.2):
    """Create train and test data loaders for each federation"""
    train_loaders, test_loaders = [], []
    
    for X, y in federations:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, stratify=y)
        except ValueError:
            # Fall back to non-stratified split if some classes are too small
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split)
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        
        train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
        test_loaders.append(DataLoader(test_dataset, batch_size=batch_size))
    
    return train_loaders, test_loaders
