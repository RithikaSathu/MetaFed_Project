# preprocessing.py - PAMAP2 Dataset Preprocessing
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
    COLUMNS = ['timestamp', 'activity_id', 'heart_rate'] + \
              [f'hand_temp', f'hand_acc_x', f'hand_acc_y', f'hand_acc_z',
               f'hand_gyro_x', f'hand_gyro_y', f'hand_gyro_z',
               f'hand_mag_x', f'hand_mag_y', f'hand_mag_z'] + \
              [f'chest_temp', f'chest_acc_x', f'chest_acc_y', f'chest_acc_z',
               f'chest_gyro_x', f'chest_gyro_y', f'chest_gyro_z',
               f'chest_mag_x', f'chest_mag_y', f'chest_mag_z'] + \
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
        print("\nNo data found! Please ensure PAMAP2 .dat files are in:")
        print(f"  {config.RAW_DATA_DIR}")
        print("\nExpected files: subject101.dat, subject102.dat, ..., subject109.dat")
        return
    
    # Create federations
    print("\nCreating federations...")
    federation_data = preprocessor.create_federations(all_data)
    
    # Save processed data
    preprocessor.save_processed_data(federation_data)
    
    print("\n" + "="*50)
    print("Preprocessing complete!")
    print("="*50)

if __name__ == '__main__':
    main()
