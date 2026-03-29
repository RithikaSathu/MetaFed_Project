import torch
import copy
import json
from tqdm import tqdm

from config import *
from models import CNN, LSTMModel, TransformerModel
from algorithms import fedavg_aggregate, fedbn_aggregate, fedprox_train, fedprox_aggregate, metafed_distill
from utils import load_pamap2_federations, create_data_loaders, train_client, evaluate

def get_model(model_name):
    models = {
        'CNN': CNN,
        'LSTM': LSTMModel,
        'Transformer': TransformerModel
    }
    return models[model_name](INPUT_SIZE, NUM_CLASSES)

def run_federated_training(model_name, algorithm, train_loaders, test_loaders, device):
    """Run federated training for a specific model and algorithm"""
    print(f"\n{'='*50}")
    print(f"Training {model_name} with {algorithm}")
    print(f"{'='*50}")
    
    num_clients = len(train_loaders)
    weights = [1.0 / num_clients] * num_clients
    
    global_model = get_model(model_name).to(device)
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    
    progress = []
    
    for round_num in tqdm(range(1, NUM_ROUNDS + 1)):
        # Local training
        for i in range(num_clients):
            if algorithm == 'fedprox':
                client_models[i] = fedprox_train(
                    client_models[i], global_model, train_loaders[i],
                    torch.optim.Adam(client_models[i].parameters(), lr=LEARNING_RATE),
                    torch.nn.CrossEntropyLoss(), device, epochs=LOCAL_EPOCHS
                )
            else:
                client_models[i] = train_client(
                    client_models[i], train_loaders[i], device, LOCAL_EPOCHS, LEARNING_RATE
                )
        
        # Aggregation
        if algorithm == 'fedavg':
            global_model = fedavg_aggregate(global_model, client_models, weights)
        elif algorithm == 'fedbn':
            global_model = fedbn_aggregate(global_model, client_models, weights)
        elif algorithm == 'fedprox':
            global_model = fedprox_aggregate(global_model, client_models, weights)
        elif algorithm == 'metafed':
            global_model = fedavg_aggregate(global_model, client_models, weights)
            client_models = metafed_distill(client_models, train_loaders, device)
        
        # Sync clients
        for i in range(num_clients):
            if algorithm != 'fedbn':
                client_models[i].load_state_dict(global_model.state_dict())
        
        # Evaluate
        if round_num in [1, 5, 10, 15, 20, 25, 30]:
            accs = [evaluate(global_model, test_loaders[i], device) for i in range(num_clients)]
            avg_acc = sum(accs) / len(accs)
            progress.append({'round': round_num, 'accuracy': avg_acc, 'per_federation': accs})
            print(f"Round {round_num}: Avg Acc = {avg_acc:.2f}%")
    
    # Final evaluation per federation
    final_accs = [evaluate(global_model, test_loaders[i], device) for i in range(num_clients)]
    
    return {
        'model': model_name,
        'algorithm': algorithm,
        'final_accuracies': final_accs,
        'progress': progress
    }

import argparse


def main():
    parser = argparse.ArgumentParser(description='Run federated training')
    parser.add_argument('--data-path', type=str, default='./PAMAP2_Dataset/Protocol',
                        help='Path to PAMAP2 Protocol folder containing subject files')
    parser.add_argument('--rounds', type=int, default=None, help='Number of global rounds')
    parser.add_argument('--epochs', type=int, default=None, help='Local epochs per client')
    parser.add_argument('--output', type=str, default='training_results.json', help='Output JSON filename')
    parser.add_argument('--max-samples-per-fed', type=int, default=None, help='Max sequences per federation (for memory/time)')
    args = parser.parse_args()

    # Override globals if provided
    if args.rounds is not None:
        NUM_ROUNDS = args.rounds
    if args.epochs is not None:
        LOCAL_EPOCHS = args.epochs
    max_samples = args.max_samples_per_fed

    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = args.data_path
    # Validate path
    import os
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Data path {data_path} not found. Please provide PAMAP2 Protocol folder.")

    federations = load_pamap2_federations(data_path, SEQUENCE_LENGTH, NUM_FEDERATIONS, max_samples_per_fed=max_samples)
    if len(federations) == 0:
        raise RuntimeError(f"No federations found in {data_path}. Check that subject files exist and are readable.")

    train_loaders, test_loaders = create_data_loaders(federations, BATCH_SIZE)

    results = []
    model_names = ['CNN', 'LSTM', 'Transformer']
    algorithms = ['fedavg', 'fedbn', 'fedprox', 'metafed']

    for model_name in model_names:
        for algo in algorithms:
            result = run_federated_training(model_name, algo, train_loaders, test_loaders, device)
            results.append(result)

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*50)
    print(f"TRAINING COMPLETE! Results saved to {args.output}")
    print("="*50)

if __name__ == "__main__":
    main()
