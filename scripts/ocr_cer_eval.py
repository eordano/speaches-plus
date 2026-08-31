#!/usr/bin/env python3
"""CER/WER gate for the dots QLoRA loop. refs.jsonl rows are dump-input
{image, text, bbox?}; each hyp file is the dots_ocr --rows --plain output
(one {"i", "text", ...} line per ref row, in order). With two hyp files the
second is the adapter arm and the report carries the deltas the shipping
decision reads. Rates are micro-averaged (total edit distance over total
reference length) and reported raw and markup-neutral; neutral is the gating
number, and its normalization is a port of nv-ocr tests/e2e_fixtures.rs::
markup_neutral so this gate and the conformance fixtures agree on what
counts as a text error."""

import argparse
import json
import sys


def strip_html_tags(s):
    out = []
    depth = 0
    for c in s:
        if c == '<':
            depth += 1
        elif c == '>':
            if depth:
                depth -= 1
                out.append(' ')
            else:
                out.append(c)
        elif not depth:
            out.append(c)
    return ''.join(out)


def is_rule_or_table_separator(s):
    t = s.strip()
    if not t:
        return False
    saw = False
    for c in t:
        if c in '-=_*':
            saw = True
        elif c not in '|:+ \t':
            return False
    return saw


def strip_leading_block_markers(s):
    t = s.lstrip()
    while True:
        before = t
        t = t.lstrip('#>')
        for m in ('- ', '* ', '+ '):
            if t.startswith(m):
                t = t[len(m):]
                break
        t = t.lstrip()
        if t == before:
            return t


def markup_neutral(s):
    words = []
    for raw in (s or '').splitlines():
        line = strip_html_tags(raw)
        if is_rule_or_table_separator(line):
            continue
        line = strip_leading_block_markers(line)
        line = ''.join(' ' if c in '*`|' else c for c in line)
        words.extend(line.split())
    return ' '.join(words)


def edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(refs, hyps, norm):
    cer_num = cer_den = wer_num = wer_den = 0
    rows = []
    for ref, hyp in zip(refs, hyps):
        r, h = norm(ref), norm(hyp)
        cd = edit_distance(r, h)
        wd = edit_distance(r.split(), h.split())
        cer_num += cd
        cer_den += len(r)
        wer_num += wd
        wer_den += len(r.split())
        rows.append({'cer': cd / len(r) if r else 0.0,
                     'wer': wd / len(r.split()) if r.split() else 0.0})
    return {'cer': cer_num / cer_den if cer_den else 0.0,
            'wer': wer_num / wer_den if wer_den else 0.0,
            'rows': rows}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('refs', help='dump-input JSONL with the corrected text')
    ap.add_argument('hyps', nargs='+',
                    help='1-2 dots_ocr --rows outputs (base [, adapter])')
    args = ap.parse_args()

    refs = [r['text'] for r in load_jsonl(args.refs)]
    report = {'n': len(refs)}
    names = ['base', 'adapter'][:len(args.hyps)]
    for name, path in zip(names, args.hyps):
        hyps = [h.get('text', '') for h in load_jsonl(path)]
        if len(hyps) != len(refs):
            sys.exit(f'{path}: {len(hyps)} hyp rows for {len(refs)} refs')
        report[name] = {
            'raw': score(refs, hyps, lambda s: ' '.join((s or '').split())),
            'neutral': score(refs, hyps, markup_neutral)}
    if len(names) == 2:
        report['delta'] = {
            k: {'cer': report['adapter'][k]['cer'] - report['base'][k]['cer'],
                'wer': report['adapter'][k]['wer'] - report['base'][k]['wer']}
            for k in ('raw', 'neutral')}
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
