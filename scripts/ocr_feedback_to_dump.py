#!/usr/bin/env python3
"""Convert the portal's /ocr-read correction store into nvk-dots-vision-dump
input JSONL. Feedback rows are {ts, page_sha, media_type, bbox, category,
model_text, corrected_text, basis}; output rows are {image, text, bbox?} with
text preferring corrected_text. Repeated corrections of the same (page, bbox)
keep only the newest. --only-corrected drops rows whose corrected text merely
confirms the model text, which is what a training set should be built from."""

import argparse
import json
import mimetypes
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('feedback_dir',
                    help='the ocr-feedback dir (feedback.jsonl + pages/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--only-corrected', action='store_true')
    args = ap.parse_args()

    fb = os.path.join(args.feedback_dir, 'feedback.jsonl')
    pages = os.path.join(args.feedback_dir, 'pages')
    latest = {}
    n_read = 0
    with open(fb) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_read += 1
            key = (row['page_sha'], json.dumps(row.get('bbox')))
            if key not in latest or row.get('ts', 0) >= latest[key].get('ts', 0):
                latest[key] = row

    out_rows = []
    missing = 0
    for row in latest.values():
        text = row.get('corrected_text') or row.get('model_text')
        if not text:
            continue
        if args.only_corrected and row.get('corrected_text') == row.get('model_text'):
            continue
        ext = mimetypes.guess_extension(row.get('media_type') or '') or '.png'
        img = os.path.abspath(os.path.join(pages, row['page_sha'] + ext))
        if not os.path.exists(img):
            missing += 1
            continue
        o = {'image': img, 'text': text}
        if row.get('bbox'):
            o['bbox'] = row['bbox']
        out_rows.append(o)

    with open(args.out, 'w') as f:
        for o in out_rows:
            f.write(json.dumps(o) + '\n')
    print(f'{len(out_rows)} rows from {n_read} feedback lines '
          f'({len(latest)} unique regions, {missing} missing page images)',
          file=sys.stderr)
    if missing:
        sys.exit(1)


if __name__ == '__main__':
    main()
