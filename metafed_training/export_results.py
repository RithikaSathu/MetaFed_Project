import json

def export_to_typescript():
    with open('training_results.json', 'r') as f:
        results = json.load(f)
    
    print("// Copy this to src/data/mockResults.ts\n")
    print("export const modelResults: ModelResult[] = [")
    
    for r in results:
        model = r['model']
        algo = r['algorithm']
        accs = r['final_accuracies']
        
        for fed_idx, acc in enumerate(accs, 1):
            if algo == 'fedavg':
                print(f'  {{ model: "{model}", federation: {fed_idx}, fedavg: {acc:.1f}, fedbn: 0, fedprox: 0, metafed: 0 }},')
    
    print("];\n")
    print("// Update the values for each algorithm accordingly")

if __name__ == "__main__":
    export_to_typescript()
