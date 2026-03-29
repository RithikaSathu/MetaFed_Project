from utils import load_pamap2_federations
from config import SEQUENCE_LENGTH, NUM_FEDERATIONS

def inspect(data_path='../backend/data/raw'):
    print('Loading federations from', data_path)
    federations = load_pamap2_federations(data_path, SEQUENCE_LENGTH, NUM_FEDERATIONS)
    print('Federations returned:', len(federations))
    for idx, (X, y) in enumerate(federations):
        print(f' Federation {idx+1}: samples={len(X)}, unique_labels={len(set(y.tolist()))}')

if __name__ == '__main__':
    inspect()
