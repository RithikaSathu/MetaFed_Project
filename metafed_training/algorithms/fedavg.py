import copy

def fedavg_aggregate(global_model, client_models, weights):
    """Simple weighted averaging of all model parameters"""
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = sum(
            weights[i] * client_models[i].state_dict()[key] 
            for i in range(len(client_models))
        )
    
    global_model.load_state_dict(global_dict)
    return global_model
