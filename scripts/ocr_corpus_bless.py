#!/usr/bin/env python3
"""Standing corpus judge for dots.ocr tolerance-class arms.

`freeze` writes the deterministic train/eval split for a corpus rows file:
per page, ids rank by sha256("<seed>:<id>") and the lowest
ceil(n_page * eval_fraction) go to eval, so every page and page class is
represented on both sides and appending rows later re-freezes a v2 split
without moving any v1 assignment that its own page census does not move.
The committed split file is the authority; the rule and seed ride inside it
for the re-freeze.

`bless` (the default) judges one arm spec with the 06.10 blessing
instrument: build the three tools via nvk.sh, dump the corpus through
nvk-dots-vision-dump (the real dump reader -- corpus rows carry extra
provenance keys the reader ignores), train an adapter on the train split
under --arm envs, score base and adapter on the eval split through the
identical serving path (dots_ocr --plain --rows, optional --serve-arm envs
applied to BOTH scoring runs), and gate on markup-neutral CER via
ocr_cer_eval's own normalization. Every GPU phase wraps in gpu-flock.py.
Child processes run with NV_*/CANDLE_*/QP_*/DOTS_* scrubbed from the
inherited environment, then only the declared arm envs applied, so a
session's stray toggles cannot contaminate either arm."""

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_cer_eval import markup_neutral, score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK = '/tmp/claude-501/nvk-gpu0.lock'
CARGO_TMP = os.environ.get('NVK_CACHE') or os.path.join(
    os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache'), 'cargo-tmp')


def load_rows(path):
    with open(path) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate row ids'
    return rows


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def freeze(args):
    rows = load_rows(args.rows)
    rank = lambda rid: hashlib.sha256(f'{args.seed}:{rid}'.encode()).hexdigest()
    pages = {}
    for r in rows:
        pages.setdefault(r['page'], []).append(r['id'])
    train, evl = [], []
    for page in sorted(pages):
        ids = sorted(pages[page], key=rank)
        k = math.ceil(len(ids) * args.eval_fraction)
        evl += sorted(ids[:k])
        train += sorted(ids[k:])
    split = {
        'seed': args.seed,
        'eval_fraction': args.eval_fraction,
        'rule': 'per page: rank ids by sha256("<seed>:<id>"), lowest '
                'ceil(n*eval_fraction) are eval',
        'rows_sha256': sha_file(args.rows),
        'train': sorted(train),
        'eval': evl,
    }
    with open(args.out, 'w') as f:
        json.dump(split, f, indent=1)
        f.write('\n')
    print(f'froze {len(train)} train / {len(evl)} eval over '
          f'{len(pages)} pages -> {args.out}')


def scrubbed_env(extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('NV_', 'CANDLE_', 'QP_', 'DOTS_'))}
    env.update(extra)
    return env


def run(cmd, log, env=None, gpu=False, check=True):
    if gpu:
        cmd = [os.path.join(REPO, 'rust/scripts/gpu-flock.py'), LOCK] + cmd
    t0 = time.time()
    with open(log, 'ab') as f:
        f.write((' '.join(map(shlex.quote, cmd)) + '\n').encode())
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=f, env=env)
    if p.returncode != 0 and check:
        tail = open(log, errors='replace').read()[-3000:]
        sys.exit(f'FAILED ({p.returncode}) after {time.time()-t0:.0f}s: '
                 f'{" ".join(cmd)}\n--- {log} tail ---\n{tail}')
    return time.time() - t0 if p.returncode == 0 else -1


def build_tools(args, logdir):
    specs = [
        ('dotsq', 'nv-train', 'metal', ['--bin', 'nvk-dots-lora-train']),
        ('dotsvision', 'nv-models', 'wgpu', ['--example', 'nvk-dots-vision-dump']),
        ('dotsgraph', 'nv-models', 'metal,wgpu', ['--example', 'dots_ocr']),
    ]
    for lane, pkg, feats, what in specs:
        if not args.skip_build:
            env = dict(os.environ, NVK_LANE=lane, NVK_PKG=pkg,
                       NVK_FEATURES=feats, NVK_SHELL='.#no-models')
            run([os.path.join(REPO, 'rust/scripts/nvk.sh'), 'build',
                 *what, '--release'],
                os.path.join(logdir, f'build-{lane}.log'), env=env)
    return {
        'train': os.path.join(CARGO_TMP, 'tgt-dotsq/release/nvk-dots-lora-train'),
        'dump': os.path.join(CARGO_TMP,
                             'tgt-dotsvision/release/examples/nvk-dots-vision-dump'),
        'eval': os.path.join(CARGO_TMP, 'tgt-dotsgraph/release/examples/dots_ocr'),
    }


def materialize(rows, ids, dst):
    by_id = {r['id']: r for r in rows}
    with open(dst, 'w') as f:
        for rid in ids:
            r = dict(by_id[rid])
            r['image'] = os.path.join(REPO, r['image'])
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return [by_id[rid] for rid in ids]


def dump_side(tools, args, side_jsonl, out_dir, n_expect, log):
    idx = os.path.join(out_dir, 'index.jsonl')
    if os.path.exists(idx) and sum(1 for l in open(idx) if l.strip()) == n_expect:
        return 'cached'
    cmd = [tools['dump'], '--dots-dir', args.dots_dir,
           '--data', side_jsonl, '--out', out_dir]
    if args.dump_max_pixels:
        cmd += ['--max-pixels', str(args.dump_max_pixels)]
    secs = run(cmd, log, env=scrubbed_env({}), gpu=True)
    return f'{secs:.0f}s'


def score_arm(tools, args, eval_jsonl, extra_env, out_jsonl, log):
    with open(out_jsonl, 'w') as f:
        cmd = [os.path.join(REPO, 'rust/scripts/gpu-flock.py'), LOCK,
               tools['eval'], '--plain', '--rows', eval_jsonl]
        with open(log, 'wb') as lf:
            p = subprocess.run(cmd, stdout=f, stderr=lf,
                               env=scrubbed_env(extra_env))
    if p.returncode != 0:
        sys.exit(f'scoring failed ({p.returncode}), see {log}')
    return [json.loads(l) for l in open(out_jsonl) if l.strip()]


def bless(args):
    arm = dict(kv.split('=', 1) for kv in args.arm)
    serve_arm = dict(kv.split('=', 1) for kv in args.serve_arm)
    serve_arm.setdefault('NV_DOTS_DIR', args.dots_dir)
    rows = load_rows(args.rows)
    split = json.load(open(args.split))
    if split['rows_sha256'] != sha_file(args.rows):
        sys.exit('split was frozen against a different rows file; re-freeze')

    stamp = time.strftime('%Y%m%d-%H%M%S')
    rundir = os.path.join(args.bless_dir, 'runs', f'{stamp}-{args.tag}')
    os.makedirs(rundir, exist_ok=True)
    tools = build_tools(args, rundir)
    for t in tools.values():
        if not os.path.exists(t):
            sys.exit(f'missing tool {t} (run without --skip-build)')

    train_jsonl = os.path.join(rundir, 'train-rows.jsonl')
    eval_jsonl = os.path.join(rundir, 'eval-rows.jsonl')
    materialize(rows, split['train'], train_jsonl)
    eval_rows = materialize(rows, split['eval'], eval_jsonl)

    key = hashlib.sha256(
        (sha_file(args.rows) + sha_file(args.split)
         + str(args.dump_max_pixels)).encode()).hexdigest()[:12]
    dumps = os.path.join(args.bless_dir, 'dumps', key)
    os.makedirs(dumps, exist_ok=True)
    for side, src in (('train', train_jsonl), ('eval', eval_jsonl)):
        n = len(split[side])
        st = dump_side(tools, args, src, os.path.join(dumps, side), n,
                       os.path.join(rundir, f'dump-{side}.log'))
        print(f'dump {side}: {n} rows [{st}]', flush=True)

    adapter = os.path.join(rundir, 'adapter')
    train_cmd = [tools['train'], '--base', args.base,
                 '--data', os.path.join(dumps, 'train'),
                 '--eval-data', os.path.join(dumps, 'eval'),
                 '--out', adapter, '--steps', str(args.steps),
                 '--seed', str(args.seed)]
    tlog = os.path.join(rundir, 'train.log')
    checkpoint_fallback = False
    tsecs = run(train_cmd, tlog, env=scrubbed_env(arm), gpu=True, check=False)
    if tsecs < 0:
        tail = open(tlog, errors='replace').read()[-3000:]
        if 'Insufficient Memory' not in tail and 'OutOfMemory' not in tail:
            sys.exit(f'training failed\n--- {tlog} tail ---\n{tail}')
        checkpoint_fallback = True
        print('train: OOM on the live-graph window backward; retrying with '
              'the bit-identical NV_DOTS_QLORA_CHECKPOINT=1 escape', flush=True)
        tsecs = run(train_cmd, tlog,
                    env=scrubbed_env(dict(arm, NV_DOTS_QLORA_CHECKPOINT='1')),
                    gpu=True)
    summary = json.loads(
        [l for l in open(os.path.join(rundir, 'train.log'))
         if l.startswith('{')][-1])
    print(f'train: {args.steps} steps in {tsecs:.0f}s, loss '
          f'{summary["losses"][0]:.4f} -> {summary["losses"][-1]:.4f}, '
          f'eval loss {summary["eval_loss_base"]:.4f} -> '
          f'{summary["eval_loss_adapter"]:.4f}', flush=True)

    hyps = {}
    for name, extra in (('base', serve_arm),
                        ('adapter', dict(serve_arm,
                                         NV_DOTS_GRAPH_LORA_DIR=adapter))):
        out = os.path.join(rundir, f'hyp-{name}.jsonl')
        hyps[name] = score_arm(tools, args, eval_jsonl, extra, out,
                               os.path.join(rundir, f'score-{name}.log'))
        assert len(hyps[name]) == len(eval_rows), f'{name}: hyp count mismatch'
        print(f'score {name}: {len(hyps[name])} rows', flush=True)

    refs = [r['text'] for r in eval_rows]
    report = {'n': len(refs)}
    for name in ('base', 'adapter'):
        texts = [h.get('text', '') for h in hyps[name]]
        report[name] = {
            'raw': score(refs, texts, lambda s: ' '.join((s or '').split())),
            'neutral': score(refs, texts, markup_neutral)}
    delta = report['adapter']['neutral']['cer'] - report['base']['neutral']['cer']
    verdict = 'PASS' if delta <= 0 else 'FAIL'

    basis = {
        'tag': args.tag, 'arm': arm, 'serve_arm': args.serve_arm,
        'steps': args.steps, 'seed': args.seed,
        'rows': args.rows, 'rows_sha256': split['rows_sha256'],
        'split': args.split, 'split_sha256': sha_file(args.split),
        'base': args.base, 'dots_dir': args.dots_dir,
        'repo_head': subprocess.run(
            ['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True).stdout.strip(),
        'repo_dirty': bool(subprocess.run(
            ['git', '-C', REPO, 'status', '--porcelain'],
            capture_output=True, text=True).stdout.strip()),
        'checkpoint_fallback': checkpoint_fallback,
        'train_summary': summary, 'rundir': rundir,
    }
    print(f'\n{"id":42} {"base":>8} {"adapter":>8} {"delta":>8}')
    for r, b, a in zip(eval_rows,
                       report['base']['neutral']['rows'],
                       report['adapter']['neutral']['rows']):
        print(f'{r["id"]:42} {b["cer"]:8.4f} {a["cer"]:8.4f} '
              f'{a["cer"]-b["cer"]:+8.4f}')
    for k in ('raw', 'neutral'):
        print(f'{k}: base CER {report["base"][k]["cer"]:.4f} '
              f'adapter CER {report["adapter"][k]["cer"]:.4f} '
              f'delta {report["adapter"][k]["cer"]-report["base"][k]["cer"]:+.4f}')
    print(f'VERDICT {verdict} (neutral CER delta {delta:+.4f}, gate: '
          f'adapter <= base)')
    with open(os.path.join(rundir, 'verdict.json'), 'w') as f:
        json.dump({'verdict': verdict, 'neutral_delta': delta,
                   'report': report, 'basis': basis}, f, indent=1)
    print(f'verdict + logs: {rundir}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd')
    fz = sub.add_parser('freeze')
    fz.add_argument('--rows', required=True)
    fz.add_argument('--out', required=True)
    fz.add_argument('--seed', default='corpus-judge-v1')
    fz.add_argument('--eval-fraction', type=float, default=0.2)
    bl = sub.add_parser('bless')
    bl.add_argument('--tag', default='default')
    bl.add_argument('--arm', action='append', default=[],
                    help='trainer env ENV=VALUE, repeatable')
    bl.add_argument('--serve-arm', action='append', default=[],
                    help='serving env ENV=VALUE for BOTH scoring runs')
    bl.add_argument('--steps', type=int, default=40)
    bl.add_argument('--seed', type=int, default=7)
    bl.add_argument('--rows',
                    default=os.path.join(REPO, 'conformance/corpus-judge/rows-v1.jsonl'))
    bl.add_argument('--split',
                    default=os.path.join(REPO, 'conformance/corpus-judge/split-v1.json'))
    bl.add_argument('--base', default=os.path.expanduser(
        '~/.cache/huggingface/hub/models--mlx-community--dots.ocr-4bit/'
        'snapshots/4ab989e403d4f8cafa5fdeede5b2290a706c2405'))
    bl.add_argument('--dots-dir', default=os.path.expanduser(
        '~/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/'
        'snapshots/c0111ce6bc07803dbc267932ffef0ae3a51dc951'))
    bl.add_argument('--bless-dir',
                    default=os.path.expanduser('~/build/studio-q4/corpus-judge'))
    bl.add_argument('--dump-max-pixels', type=int, default=None)
    bl.add_argument('--skip-build', action='store_true')
    argv = sys.argv[1:]
    if not argv or argv[0] not in ('freeze', 'bless'):
        argv = ['bless'] + argv
    args = ap.parse_args(argv)
    if args.cmd == 'freeze':
        freeze(args)
    else:
        bless(args)


if __name__ == '__main__':
    main()
