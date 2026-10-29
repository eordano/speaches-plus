import argparse
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

ENCODER = 'Qwen/Qwen3-8B'
ENCODER_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
HEAD = 'Contrastive-LM/CLM-v0.1-8B'
HEAD_REVISION = 'e939398d4556fcd9400c76fa8c5a513202f42b0a'
UPSTREAM_REVISION = 'bb42c6c5bf914fd449bed2f6ca65be80602cb1f7'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    encoder = snapshot_download(ENCODER, revision=ENCODER_REVISION,
        allow_patterns=['*.safetensors', '*.json', '*.txt', '*.jinja'])
    head = hf_hub_download(HEAD, 'CLM_v0.1-8B.pt', revision=HEAD_REVISION)
    manifest = {'encoder': encoder, 'encoder_id': ENCODER, 'encoder_revision': ENCODER_REVISION,
                'head': head, 'head_id': HEAD, 'head_revision': HEAD_REVISION,
                'upstream_revision': UPSTREAM_REVISION}
    (args.out / 'models.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)

if __name__ == '__main__':
    main()
