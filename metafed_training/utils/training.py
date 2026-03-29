import torch
import torch.nn as nn
import copy

def train_client(model, train_loader, device, epochs=5, lr=0.001):
    """Standard local training for a client"""
    model = copy.deepcopy(model)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    return model

def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100.0 * correct / total
