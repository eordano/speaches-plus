import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re

INSTRUCTIONS = ("A voice assistant just heard the user's latest message. "
                "Which tool should it call first to handle it? Choose none "
                "if it should just answer.")
NONE_DESC = 'No tool: answer or chat directly from what is already known.'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()

def canonical(row):
    row = copy.deepcopy(row)
    for qid, q in row['questions'].items():
        if q.get('type') not in {'choice', 'noul', 'score'}:
            raise ValueError('question type must be choice, noul, or score')
        if qid == 'tool' and q['type'] == 'choice':
            criteria = q['criteria']
            if 'none' in criteria and 'no_tool' in criteria:
                raise ValueError('ambiguous no-call options')
            if 'no_tool' in criteria:
                criteria['none'] = criteria.pop('no_tool')
            if 'none' not in criteria:
                raise ValueError('tool choice needs none or no_tool')
            criteria['none'] = NONE_DESC
            q['instructions'] = INSTRUCTIONS
            if q['label'] == 'no_tool':
                q['label'] = 'none'
            if any(not isinstance(v, (str, type(None))) for v in criteria.values()):
                raise ValueError('tool descriptions must be strings or null')
            q['criteria'] = {k: (v if v == k or v.startswith(k + ': ') else f'{k}: {v}') if v else k
                             for k, v in criteria.items()}
        kind = q['type']
        if kind == 'score' and len(q['criteria']) < 2:
            raise ValueError('score needs at least two levels')
        keys = list(q['criteria']) if kind == 'choice' else (
            [str(i) for i in range(len(q['criteria']))] if kind == 'score' else ['false', 'true'])
        label = str(q['label']).lower() if isinstance(q['label'], bool) else str(q['label'])
        if label not in keys or not 1 <= len(keys) <= 255:
            raise ValueError(f'{qid}: label not in offered choices or invalid number of choices')
        q['label'] = label
    return row

def read(path):
    with open(path) as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield canonical(json.loads(line))
                except Exception as exc:
                    raise ValueError(f'{path}:{line_no}: {exc}') from exc

def identity(row):
    return digest([row['state'], {k: {a: b for a, b in q.items() if a != 'label'}
                                for k, q in row['questions'].items()}])

def question_group(row):
    state = row['state']
    if isinstance(state, str):
        latest = re.findall(r'(?:^|\n)(?:User|Latest user message|latest user):\s*([^\n]+)', state)
        if latest:
            return digest(' '.join(latest[-1].split()).casefold())
    return digest(state)

def prepare(sources, exclusions, output):
    forbidden = {question_group(row) for path in exclusions for row in read(path)}
    unique, conflicts, counts = {}, set(), Counter()
    forbidden_conversations = set()
    for path in sources:
        for row in read(path):
            key = identity(row)
            labels = {qid: q['label'] for qid, q in row['questions'].items()}
            if key in unique:
                if labels != {qid: q['label'] for qid, q in unique[key]['questions'].items()}:
                    conflicts.add(key)
                counts['duplicate'] += 1
                continue
            if question_group(row) in forbidden:
                meta = row.get('_meta') or {}
                if meta.get('group'):
                    forbidden_conversations.add((meta.get('source', '?'), meta['group']))
                counts['heldout_question_excluded'] += 1
                continue
            unique[key] = row
    unique = {key: row for key, row in unique.items() if (
        (row.get('_meta') or {}).get('source', '?'),
        (row.get('_meta') or {}).get('group')) not in forbidden_conversations}
    groups = {}
    parent = {}
    def root(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = root(parent[key])
        return parent[key]
    for key, row in unique.items():
        group = question_group(row)
        meta = row.get('_meta') or {}
        conversation = (meta.get('source', '?'), meta.get('group'))
        if conversation[1]:
            previous = groups.setdefault(conversation, group)
            parent[root(group)] = root(previous)
    splits = {s: [] for s in ('train', 'validation', 'calibration', 'test')}
    for key, row in unique.items():
        if key in conflicts:
            continue
        value = int(root(question_group(row))[:8], 16) % 1000
        split = 'test' if value < 20 else 'calibration' if value < 40 else 'validation' if value < 60 else 'train'
        splits[split].append(row)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'sources': [str(p) for p in sources], 'excluded': [str(p) for p in exclusions],
                'conflicting_decisions_removed': len(conflicts), 'filter_counts': dict(counts), 'splits': {}}
    for split, rows in splits.items():
        path = output / (split + '.jsonl')
        with path.open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=True) + '\n')
        manifest['splits'][split] = {'rows': len(rows), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'sources': dict(Counter((r.get('_meta') or {}).get('source', '?') for r in rows)),
            'tool_labels': dict(Counter(r['questions']['tool']['label'] for r in rows if 'tool' in r['questions']))}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest

def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, action='append', required=True)
    parser.add_argument('--exclude', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.exclude, args.out)
    print(json.dumps({k: v['rows'] for k, v in result['splits'].items()}), flush=True)

if __name__ == '__main__':
    main()
