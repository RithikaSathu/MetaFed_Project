# trainer.py - Training orchestration
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
        
        print(f"\n{'='*60}")
        print(f"Starting Federated Learning Training")
        print(f"Algorithm: {self.algorithm_name.upper()}")
        print(f"Model: {self.model_name.upper()}")
        print(f"Device: {self.device}")
        print(f"Rounds: {num_rounds}")
        print(f"{'='*60}\n")
        
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
        
        print(f"\nTraining Complete!")
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
        
        print(f"\n{'='*60}")
        print(f"MetaFed Heterogeneous Training")
        print(f"Model Assignment: {self.model_assignment}")
        print(f"{'='*60}\n")
        
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


# run_all_experiments.py - Run all FL algorithms and compare
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
    print("\n" + "="*60)
    print("RUNNING HOMOGENEOUS FL EXPERIMENTS")
    print("="*60)
    
    for algo in algorithms:
        print(f"\n>>> Training {algo.upper()} with {model.upper()}")
        
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
    print("\n" + "="*60)
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
    
    print(f"\nAll results saved to {comparison_file}")
    
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
