# algorithms.py - Federated Learning Algorithms
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
