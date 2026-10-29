import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from clm_download import UPSTREAM_REVISION
from clm_prepare import prepare

def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cccp-train', type=Path, required=True)
    parser.add_argument('--cccp-eval', type=Path, required=True)
    parser.add_argument('--portal', type=Path, required=True)
    parser.add_argument('--when2call-eval', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    lock = open(args.work / 'training.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scripts = Path(__file__).resolve().parent
    def run(script, *command):
        subprocess.run([sys.executable, str(scripts / script), *map(str, command)], check=True)
    upstream = args.work / 'upstream'
    if not (upstream / '.git').exists():
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/Contrastive-LM/CLM', str(upstream)], check=True)
        subprocess.run(['git', '-C', str(upstream), 'checkout', '--detach', UPSTREAM_REVISION], check=True)
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f'CLM checkout must be {UPSTREAM_REVISION}, got {revision}')
    os.environ['PYTHONPATH'] = str(upstream / 'src')
    run('clm_download.py', '--out', args.work)
    run('clm_public_data.py', '--out', args.work / 'public')
    replay_path = args.work / 'replay.jsonl'
    replay_records = 0
    with replay_path.open('w') as out, args.replay.open() as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                if 'tool' not in row['questions']:
                    out.write(line.rstrip() + '\n')
                    replay_records += 1
    print(json.dumps({'event': 'replay', 'records': replay_records,
                      'source': str(args.replay)}), flush=True)
    sources = [args.cccp_train, args.portal / 'train.jsonl', replay_path,
               *sorted((args.work / 'public').glob('*.jsonl'))]
    excluded = [args.cccp_eval, args.portal / 'development.jsonl', args.when2call_eval]
    manifest = prepare(sources, excluded, args.work / 'data')
    print(json.dumps({'event': 'prepared', 'splits': {k: v['rows'] for k, v in manifest['splits'].items()}}), flush=True)
    command = ['--models', args.work / 'models.json', '--data', args.work / 'data',
               '--cache', args.work / 'embeddings-8192.sqlite', '--out', args.work / 'runs' / 'clm-tooluse-8b-v1',
               '--epochs', args.epochs]
    run('clm_train.py', *command)

if __name__ == '__main__':
    main()
