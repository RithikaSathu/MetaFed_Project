import os
import numpy as np
from config import NUM_CLASSES


def generate(dataset_dir='PAMAP2_Dataset/Protocol', rows_per_subject=1000, features=52, learnable=True):
    os.makedirs(dataset_dir, exist_ok=True)

    for subject_id in range(1, 10):
        file_path = os.path.join(dataset_dir, f'subject10{subject_id}.dat')
        with open(file_path, 'w') as f:
            # Create labels balanced across classes
            labels = np.random.choice(range(1, NUM_CLASSES + 1), size=rows_per_subject)
            # If learnable, create class-dependent feature means so models can learn
            if learnable:
                feats = np.zeros((rows_per_subject, features), dtype=np.float32)
                for i, lbl in enumerate(labels):
                    center = (lbl % features) * 0.1  # small offset per class
                    feats[i] = np.random.randn(features).astype(np.float32) + center
            else:
                feats = np.random.randn(rows_per_subject, features).astype(np.float32)

            for i in range(rows_per_subject):
                # Columns: timestamp, label, placeholder, placeholder, then features
                cols = [f"{i:.3f}", str(int(labels[i])), "0.0", "0.0"] + [f"{x:.6f}" for x in feats[i]]
                f.write(' '.join(cols) + '\n')


if __name__ == '__main__':
    generate()
