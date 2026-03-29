def fedbn_aggregate(global_model, client_models, weights):
    """Aggregate all layers EXCEPT batch normalization layers"""
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        if 'bn' not in key.lower() and 'batch' not in key.lower():
            global_dict[key] = sum(
                weights[i] * client_models[i].state_dict()[key] 
                for i in range(len(client_models))
            )
    
    global_model.load_state_dict(global_dict)
    return global_model
