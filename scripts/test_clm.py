import json
from pathlib import Path
import tempfile
import unittest

import torch
from clm_prepare import canonical, prepare, question_group
from clm_train import contrastive_loss, load_heads, export, calibrate
from clm_public_data import when2call, xlam_irrelevance, toolace, glaive, hermes
from clm.heads import make_head
from clm.schema import build_pairs

def row(state, label='run', group=None):
    return {'state': 'User: ' + state, '_meta': {'source': 'fixture', 'group': group},
            'questions': {'tool': {'type': 'choice', 'instructions': 'Pick',
                'criteria': {'no_tool': None, 'run': 'Execute a command'}, 'label': label}}}

class DataTest(unittest.TestCase):
    def test_public_formats_produce_the_expected_upstream_actions(self):
        tools = [{'name': 'weather', 'description': 'Look up the weather'}]
        cases = [
            (when2call, {'tools': tools, 'messages': [
                {'role': 'user', 'content': 'Weather?'},
                {'role': 'assistant', 'content': 'TOOLCALL [{"name":"weather"}]'}]}),
            (xlam_irrelevance, {'tools': json.dumps(tools), 'query': 'Weather?',
                               'answers': '[{"name":"weather"}]'}),
            (toolace, {'system': 'invoke: ' + json.dumps(tools), 'conversations': [
                {'from': 'user', 'value': 'Weather?'},
                {'from': 'assistant', 'value': '[weather(city="Berlin")]'}]}),
            (glaive, {'system': json.dumps(tools),
                      'chat': 'USER: Weather? ASSISTANT: <functioncall> {"name":"weather"}'}),
            (hermes, {'tools': tools, 'conversations': [
                {'from': 'human', 'value': 'Weather?'},
                {'from': 'gpt', 'value': '<tool_call>{"name":"weather"}</tool_call>'}]}),
        ]
        for parse, source in cases:
            with self.subTest(source=parse.__name__):
                result, = [canonical(r) for r in parse([source]) if r]
                q = result['questions']['tool']
                state, keys, actions = build_pairs(result['state'], result['questions'])['tool']
                self.assertEqual(q['label'], 'weather')
                self.assertEqual(actions[keys.index(q['label'])], 'weather: Look up the weather')
                self.assertEqual(state.split('\n\n')[0], 'User: Weather?')

    def test_glaive_keeps_visible_tool_results_and_drops_unparseable_calls(self):
        source = {'system': '[{"name":"weather"}]', 'chat':
            'USER: Weather? ASSISTANT: <functioncall> {"name":"weather"} '
            'FUNCTION RESPONSE: Sunny USER: Thanks ASSISTANT: You are welcome'}
        first, last = [canonical(r) for r in glaive([source]) if r]
        self.assertEqual(first['questions']['tool']['label'], 'weather')
        self.assertNotIn('Sunny', first['state'])
        self.assertIn('Tool: Sunny', last['state'])
        self.assertNotIn('You are welcome', last['state'])
        self.assertEqual(last['questions']['tool']['label'], 'none')
        broken = {'system': '[{"name":"weather"}]',
                  'chat': 'USER: Weather? ASSISTANT: <functioncall> {broken'}
        self.assertFalse(any(glaive([broken])))

    def test_portal_rows_canonicalize_and_reject_unknown_labels(self):
        source = row('hello', 'no_tool')
        result = canonical(source)
        self.assertEqual(result['questions']['tool']['label'], 'none')
        self.assertEqual(result['questions']['tool']['criteria']['run'], 'run: Execute a command')
        self.assertEqual(source['questions']['tool']['label'], 'no_tool')
        self.assertEqual(canonical(result), result)
        with self.assertRaises(ValueError):
            canonical(row('hello', 'missing'))

    def test_group_splits_and_holdout_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'source.jsonl'
            heldout = root / 'heldout.jsonl'
            rows = [row(f'question {i}', group=str(i // 4)) for i in range(300)]
            source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            heldout.write_text(json.dumps(rows[0]) + '\n')
            manifest = prepare([source], [heldout], root / 'out')
            assignments = {}
            for split in manifest['splits']:
                for line in (root / 'out' / (split + '.jsonl')).read_text().splitlines():
                    got = json.loads(line)
                    self.assertNotEqual(question_group(got), question_group(rows[0]))
                    self.assertNotEqual(got['_meta']['group'], '0')
                    group = got['_meta']['group']
                    self.assertEqual(assignments.setdefault(group, split), split)
            self.assertEqual(manifest['filter_counts']['heldout_question_excluded'], 1)

class TrainingTest(unittest.TestCase):
    def test_contrastive_loss_trains_heads_exports_and_calibrates(self):
        logits = torch.tensor([[9., -9.], [9., -9.], [-9., 9.]], requires_grad=True)
        targets = torch.tensor([0, 0, 1])
        loss = contrastive_loss(logits, targets)
        expected = torch.log(torch.tensor(2.)) / 4
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

        torch.manual_seed(1)
        cfg = {'width': 12, 'depth': 3, 'projection_dim': 5, 'hidden_size': 8,
               'activation': 'gelu', 'layernorm': True, 'residual': True}
        kw = {'width': 12, 'depth': 3, 'proj': 5, 'hidden': 8, 'layernorm': True, 'residual': True}
        ck = {'cfg': cfg, 'state_head': make_head(**kw).state_dict(),
              'action_head': make_head(**kw).state_dict(), 'logit_scale': torch.tensor(1.)}
        sh, ah, scale = load_heads(ck, 'cpu')
        x = torch.randn(12, 8)
        optimizer = torch.optim.Adam([*sh.parameters(), *ah.parameters(), scale], lr=.01)
        losses = []
        for _ in range(15):
            a = torch.nn.functional.normalize(sh(x), dim=-1)
            b = torch.nn.functional.normalize(ah(x), dim=-1)
            loss = contrastive_loss(scale.exp() * a @ b.T, torch.arange(len(x)))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertLess(losses[-1], losses[0])
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            export(ck, {'encoder': '/fixture', 'encoder_id': 'fixture', 'encoder_revision': 'fixture'}, out, 'fixture', 8192, 1.5)
            meta = json.loads((out / 'clm.json').read_text())
            self.assertEqual(meta['temperature'], 1.5)
            self.assertTrue((out / 'clm_head.safetensors').is_file())
        rows = [{'logits': [8., -8.], 'label': i % 2} for i in range(20)]
        self.assertGreater(calibrate(rows), 1.0)

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
