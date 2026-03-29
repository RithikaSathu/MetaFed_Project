import time
import os
import argparse
import subprocess

def wait_for_file(path, poll=10, timeout=None):
    start = time.time()
    while True:
        if os.path.exists(path):
            return True
        if timeout and (time.time() - start) > timeout:
            return False
        print(f'Waiting for {path}...')
        time.sleep(poll)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='training_results_real.json')
    parser.add_argument('--out', default='plots/training_results_real.png')
    parser.add_argument('--poll', type=int, default=10)
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    args = parser.parse_args()

    found = wait_for_file(args.json, poll=args.poll, timeout=args.timeout)
    if not found:
        print(f'File {args.json} not found within timeout; exiting.')
        return

    print(f'Found {args.json}; generating plot...')
    # Run the plotting script
    cmd = [os.path.join('.', 'venv', 'Scripts', 'python'), 'scripts/plot_training_results.py', '--json', args.json, '--out', args.out]
    try:
        subprocess.check_call(cmd)
        print('Plot generation complete.')
    except subprocess.CalledProcessError as e:
        print('Plotting failed:', e)

if __name__ == '__main__':
    main()
