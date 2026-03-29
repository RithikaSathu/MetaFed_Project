import os

DATA_DIR = '../backend/data/raw'

def check_subject(subj_file):
    path = os.path.join(os.path.dirname(__file__), '..', DATA_DIR)
    path = os.path.normpath(path)
    file_path = os.path.join(path, subj_file)
    print('Checking', file_path)
    if not os.path.exists(file_path):
        print('  Not found')
        return
    cnt = 0
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 20:
                break
            parts = line.strip().split()
            print('  parts', len(parts))
            cnt += 1
    print(f'  Read {cnt} lines (preview)')

if __name__ == '__main__':
    for i in range(1, 10):
        fname = f'subject10{i}.dat'
        check_subject(fname)
