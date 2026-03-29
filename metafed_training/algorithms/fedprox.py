import torch
import copy

def fedprox_train(model, global_model, train_loader, optimizer, criterion, device, mu=0.01, epochs=5):
    """Training with proximal term to stay close to global model"""
    model.train()
    global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}
    
    for _ in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Add proximal term
            proximal = 0
            for name, param in model.named_parameters():
                proximal += ((param - global_weights[name].to(device)) ** 2).sum()
            loss += (mu / 2) * proximal
            
            loss.backward()
            optimizer.step()
    
    return model

def fedprox_aggregate(global_model, client_models, weights):
    """Same as FedAvg aggregation"""
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = sum(
            weights[i] * client_models[i].state_dict()[key] 
            for i in range(len(client_models))
        )
    
    global_model.load_state_dict(global_dict)
    return global_model
