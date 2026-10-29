import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import torch
import torch.nn.functional as F
from clm.heads import make_head
from clm.schema import build_pairs

def log(event, **fields):
    print(json.dumps({'event': event, **fields}), flush=True)

def examples(path):
    result = []
    with open(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for qid, (state, keys, actions) in build_pairs(row['state'], row['questions']).items():
                q = row['questions'][qid]
                label = str(q['label']).lower() if isinstance(q['label'], bool) else str(q['label'])
                if len(set(actions)) != len(actions):
                    raise ValueError(f'{path}: indistinguishable action texts in {qid}')
                result.append({'state': state, 'actions': actions, 'keys': keys, 'label': keys.index(label),
                               'source': (row.get('_meta') or {}).get('source', '?'), 'qid': qid})
    if not result:
        raise ValueError(f'empty dataset: {path}')
    return result

def embeddings(texts, models, cache_path, max_tokens, batch_tokens):
    from transformers import AutoModel, AutoTokenizer
    recipe = {'encoder_revision': models['encoder_revision'], 'pooling': 'last',
              'max_tokens': max_tokens, 'truncate': 'tail', 'add_special_tokens': False}
    db = sqlite3.connect(cache_path)
    db.execute('CREATE TABLE IF NOT EXISTS metadata (recipe TEXT NOT NULL)')
    expected = json.dumps(recipe, sort_keys=True)
    stored = db.execute('SELECT recipe FROM metadata').fetchone()
    if stored and stored[0] != expected:
        raise ValueError('embedding cache belongs to a different encoder or token recipe')
    if not stored:
        db.execute('INSERT INTO metadata VALUES (?)', (expected,))
    db.execute('CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB, tokens INT, original_tokens INT)')
    db.commit()
    keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
    existing = {r[0] for r in db.execute('SELECT key FROM embeddings')}
    missing = [i for i, key in enumerate(keys) if key not in existing]
    log('embedding_cache', texts=len(texts), missing=len(missing), recipe=recipe)
    if missing:
        tokenizer = AutoTokenizer.from_pretrained(models['encoder'], local_files_only=True)
        sequences = {}
        originals = {}
        for index in missing:
            ids = tokenizer.encode(texts[index], add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(' ', add_special_tokens=False)
            originals[index] = len(ids)
            sequences[index] = ids[-(max_tokens - 1):]
        missing.sort(key=lambda i: len(sequences[i]))
        model = AutoModel.from_pretrained(models['encoder'], dtype=torch.bfloat16,
            attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
        model.requires_grad_(False)
        count, started, position = 0, time.monotonic(), 0
        with torch.inference_mode():
            while position < len(missing):
                end = position + 1
                while end < len(missing) and end - position < 32 and (end - position + 1) * len(sequences[missing[end]]) <= batch_tokens:
                    end += 1
                indexes = missing[position:end]
                lengths = [len(sequences[i]) for i in indexes]
                width = max(lengths)
                ids = torch.full((len(indexes), width), tokenizer.pad_token_id or 0, dtype=torch.long, device='cuda')
                mask = torch.zeros_like(ids)
                for j, i in enumerate(indexes):
                    ids[j, :lengths[j]] = torch.tensor(sequences[i], device='cuda')
                    mask[j, :lengths[j]] = 1
                hidden = model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                vectors = F.normalize(hidden[torch.arange(len(indexes), device='cuda'),
                                             torch.tensor(lengths, device='cuda') - 1].float(), dim=-1)
                vectors = vectors.cpu().numpy().astype(np.float16)
                db.executemany('INSERT INTO embeddings VALUES (?, ?, ?, ?)',
                    [(keys[i], vector.tobytes(), length, originals[i]) for i, vector, length in zip(indexes, vectors, lengths)])
                db.commit()
                count += len(indexes)
                if position == 0 or count // 256 != (count - len(indexes)) // 256 or end == len(missing):
                    log('embedding', done=count, total=len(missing), max_batch_length=width,
                        seconds=round(time.monotonic() - started, 1))
                position = end
        del model, hidden, vectors, ids, mask
        gc.collect()
        torch.cuda.empty_cache()
    vectors, lengths, originals = [], [], []
    for key in keys:
        vector, length, original = db.execute('SELECT vector,tokens,original_tokens FROM embeddings WHERE key=?', (key,)).fetchone()
        vectors.append(np.frombuffer(vector, dtype=np.float16))
        lengths.append(length)
        originals.append(original)
    db.close()
    return torch.from_numpy(np.stack(vectors)), lengths, originals

def load_heads(checkpoint, device):
    cfg = checkpoint['cfg']
    kw = {'width': cfg['width'], 'depth': cfg['depth'], 'proj': cfg.get('projection_dim', 512),
          'activation': cfg.get('activation', 'gelu'), 'layernorm': cfg.get('layernorm', False),
          'residual': cfg.get('residual', False), 'hidden': cfg.get('hidden_size', 4096)}
    state, action = make_head(**kw).to(device), make_head(**kw).to(device)
    state.load_state_dict(checkpoint['state_head'])
    action.load_state_dict(checkpoint['action_head'])
    scale = torch.nn.Parameter(torch.as_tensor(checkpoint['logit_scale']).float().to(device).clone())
    return state, action, scale

def batch_logits(batch, vectors, sh, ah, scale, device):
    pool = sorted({i for row in batch for i in row['action_ids']})
    column = {value: index for index, value in enumerate(pool)}
    sz = F.normalize(sh(vectors[[r['state_id'] for r in batch]].to(device).float()), dim=-1)
    az = F.normalize(ah(vectors[pool].to(device).float()), dim=-1)
    logits = scale.exp().clamp(max=100) * sz @ az.T
    targets = torch.tensor([column[r['action_ids'][r['label']]] for r in batch], device=device)
    return logits, targets, column

def contrastive_loss(logits, targets):
    forward = F.cross_entropy(logits, targets)
    positive = F.one_hot(targets, logits.shape[1]).float().T
    keep = positive.sum(-1) > 0
    positive = positive[keep] / positive[keep].sum(-1, keepdim=True)
    backward = -(positive * F.log_softmax(logits.T[keep], dim=-1)).sum(-1).mean()
    return (forward + backward) / 2

@torch.no_grad()
def evaluate(rows, vectors, sh, ah, scale, device, temperature=1.0):
    output = []
    for start in range(0, len(rows), 128):
        batch = rows[start:start + 128]
        logits, _, columns = batch_logits(batch, vectors, sh, ah, scale, device)
        for i, row in enumerate(batch):
            values = logits[i, [columns[a] for a in row['action_ids']]].cpu().double() / temperature
            probs = values.softmax(-1).tolist()
            output.append({'p': probs, 'logits': values.tolist(), 'label': row['label'],
                           'keys': row['keys'], 'source': row['source'], 'qid': row['qid']})
    return output

def metrics(rows):
    n = len(rows)
    picks = [int(np.argmax(r['p'])) for r in rows]
    hits = [p == r['label'] for p, r in zip(picks, rows)]
    confidence = [max(r['p']) for r in rows]
    ece = sum(abs(sum(hits[i] for i in ids) - sum(confidence[i] for i in ids)) for ids in (
        [i for i, p in enumerate(confidence) if min(int(p * 15), 14) == band] for band in range(15))) / n
    tools = [(p, r) for p, r in zip(picks, rows) if r['qid'] == 'tool' and r['keys'][r['label']] not in {'none', 'no_tool'}]
    order = sorted(range(n), key=lambda i: -confidence[i])
    errors, coverage = 0, 0
    for j, i in enumerate(order, 1):
        errors += not hits[i]
        if (j == n or confidence[order[j]] != confidence[i]) and errors / j <= .05:
            coverage = j / n
    return {'n': n, 'acc': sum(hits) / n,
        'nll': -sum(math.log(max(r['p'][r['label']], 1e-12)) for r in rows) / n,
        'brier': sum(sum((p - (i == r['label'])) ** 2 for i, p in enumerate(r['p'])) for r in rows) / n,
        'ece_15_bins': ece, 'cov_at_5pct_error': coverage,
        'which_tool': sum(p == r['label'] for p, r in tools) / len(tools) if tools else None,
        'tool_n': len(tools)}

def calibrate(rows):
    width = max(len(r['logits']) for r in rows)
    logits = torch.tensor([r['logits'] + [-1e9] * (width - len(r['logits'])) for r in rows], dtype=torch.float64)
    labels = torch.tensor([r['label'] for r in rows])
    value = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([value], max_iter=50, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / value.exp().clamp(.05, 20), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return value.exp().clamp(.05, 20).item()

def export(checkpoint, models, out, name, max_tokens, temperature):
    from safetensors.torch import save_file
    tensors = {f'{prefix}.{key}': value.detach().cpu().float().contiguous()
               for prefix in ('state_head', 'action_head') for key, value in checkpoint[prefix].items()}
    save_file(tensors, str(out / 'clm_head.safetensors'))
    meta = {'name': name, 'encoder_dir': models['encoder'], 'encoder_id': models['encoder_id'],
            'encoder_revision': models['encoder_revision'], 'max_tokens': max_tokens,
            'temperature': temperature, 'logit_scale': float(checkpoint['logit_scale']),
            'cfg': checkpoint['cfg'], 'format': 1}
    (out / 'clm.json').write_text(json.dumps(meta, indent=2) + '\n')
    sh, ah, _ = load_heads(checkpoint, 'cpu')
    raw = F.normalize(torch.randn(3, meta['cfg'].get('hidden_size', 4096), generator=torch.Generator().manual_seed(82)), dim=-1)
    with torch.no_grad():
        oracle = {'input': raw.tolist(), 'state': F.normalize(sh(raw), dim=-1).tolist(),
                  'action': F.normalize(ah(raw), dim=-1).tolist()}
    (out / 'head_oracle.json').write_text(json.dumps(oracle) + '\n')

def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--name', default='clm-tooluse-8b-v1')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--embed-batch-tokens', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    if args.max_tokens < 2 or args.batch < 2 or args.epochs < 1 or args.lr <= 0:
        parser.error('need max-tokens >= 2, batch >= 2, epochs >= 1, and a positive learning rate')
    torch.set_num_threads(8)
    torch.manual_seed(42)
    free, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(.8 * free / total)
    log('gpu_budget', allocator_limit_bytes=int(.8 * free), free_bytes=free)
    args.out.mkdir(parents=True, exist_ok=True)
    models = json.loads(args.models.read_text())
    data = {s: examples(args.data / f'{s}.jsonl') for s in ('train', 'validation', 'calibration', 'test')}
    texts = list(dict.fromkeys(t for rows in data.values() for row in rows for t in (row['state'], *row['actions'])))
    text_ids = {t: i for i, t in enumerate(texts)}
    vectors, lengths, original_lengths = embeddings(texts, models, args.cache, args.max_tokens, args.embed_batch_tokens)
    for rows in data.values():
        for row in rows:
            row['state_id'] = text_ids[row['state']]
            row['action_ids'] = [text_ids[a] for a in row['actions']]
    histogram = {}
    for split, rows in data.items():
        histogram[split] = {'states': len(rows), 'truncated': sum(original_lengths[r['state_id']] >= args.max_tokens for r in rows),
            'bands': dict(Counter(next(str(b) for b in (512, 1024, 2048, 4096, 6144, args.max_tokens) if lengths[r['state_id']] <= b) for r in rows))}
    (args.out / 'context_lengths.json').write_text(json.dumps(histogram, indent=2) + '\n')
    log('context_lengths', **histogram)
    ck = torch.load(models['head'], map_location='cpu', weights_only=True)
    device = 'cuda'
    sh, ah, scale = load_heads(ck, device)
    optimizer = torch.optim.AdamW([*sh.parameters(), *ah.parameters(), scale], lr=args.lr)
    baseline = {s: metrics(evaluate(data[s], vectors, sh, ah, scale, device)) for s in ('validation', 'test')}
    log('baseline', **baseline)
    (args.out / 'baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
    best = baseline['validation']['nll']
    torch.save(ck, args.out / 'best_head.pt')
    history = []
    for epoch in range(1, args.epochs + 1):
        order = list(data['train'])
        random.Random(42 + epoch).shuffle(order)
        total, count = 0.0, 0
        for start in range(0, len(order), args.batch):
            batch = order[start:start + args.batch]
            logits, targets, _ = batch_logits(batch, vectors, sh, ah, scale, device)
            loss = contrastive_loss(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError('non-finite contrastive loss')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([*sh.parameters(), *ah.parameters(), scale], 1)
            optimizer.step()
            total += loss.item() * len(batch)
            count += len(batch)
            if start // args.batch % 25 == 0:
                log('train', epoch=epoch, decisions=count, total=len(order), loss=loss.item())
        validation = metrics(evaluate(data['validation'], vectors, sh, ah, scale, device))
        item = {'epoch': epoch, 'loss': total / count, 'validation': validation}
        history.append(item)
        log('epoch', **item)
        if validation['nll'] < best:
            best = validation['nll']
            ck = {'cfg': ck['cfg'], 'logit_scale': scale.detach().cpu(),
                  'state_head': {k: v.detach().cpu() for k, v in sh.state_dict().items()},
                  'action_head': {k: v.detach().cpu() for k, v in ah.state_dict().items()}}
            torch.save(ck, args.out / 'best_head.pt')
        (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    ck = torch.load(args.out / 'best_head.pt', map_location='cpu', weights_only=True)
    del sh, ah, scale, optimizer
    sh, ah, scale = load_heads(ck, device)
    temperature = calibrate(evaluate(data['calibration'], vectors, sh, ah, scale, device))
    rows = evaluate(data['test'], vectors, sh, ah, scale, device, temperature)
    report = {'temperature': temperature, 'metrics': metrics(rows), 'baseline': baseline,
              'by_source': {s: metrics([r for r in rows if r['source'] == s]) for s in sorted({r['source'] for r in rows})}}
    (args.out / 'test.json').write_text(json.dumps({'report': report, 'rows': rows}) + '\n')
    export(ck, models, args.out, args.name, args.max_tokens, temperature)
    log('complete', **report)
    (args.out / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')

if __name__ == '__main__':
    main()
