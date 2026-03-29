import json
import os
from collections import defaultdict

def summarize_and_plot(json_path='training_results_real.json', out_path='plots/training_results_real.png'):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file {json_path} not found.")

    with open(json_path, 'r') as f:
        results = json.load(f)

    # Summarize final accuracies
    summary = defaultdict(dict)
    for r in results:
        key = (r['model'], r['algorithm'])
        final_mean = sum(r.get('final_accuracies', [])) / max(1, len(r.get('final_accuracies', [])))
        summary[r['model']][r['algorithm']] = final_mean

    print('Final mean accuracies (model -> algorithm -> mean%):')
    for model, algs in summary.items():
        print(' ', model)
        for alg, val in algs.items():
            print(f"    {alg}: {val:.2f}%")

    # Try plotting if matplotlib is available
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('\nmatplotlib not available; skipping plot generation.')
        return

    # Bar plot of mean final accuracies
    labels = []
    values = []
    for model in sorted(summary.keys()):
        for alg in sorted(summary[model].keys()):
            labels.append(f"{model}-{alg}")
            values.append(summary[model][alg])

    plt.figure(figsize=(max(6, len(labels) * 0.8), 4))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Final Accuracy (%)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='training_results_real.json')
    parser.add_argument('--out', default='plots/training_results_real.png')
    args = parser.parse_args()
    summarize_and_plot(args.json, args.out)
