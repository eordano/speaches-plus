import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "kev_context_mix", Path(__file__).with_name("kev_context_mix.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CharacterTokenizer:
    def __call__(self, value, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in value])

    def decode(self, ids, **_):
        return "".join(chr(value) for value in ids)


class ContextMixTest(unittest.TestCase):
    def test_balances_lengths_without_losing_labels_or_latest_user(self):
        records = []
        for index in range(600):
            agent = "claude" if index % 2 else "codex"
            label = "no_tool" if index % 5 == 0 else "Bash"
            records.append({
                "state": f"agent: {agent}\nlatest user: request {index}\nrecent:\n"
                         + "history " * 1100 + f"decision {index}",
                "questions": {"tool": {"label": label}},
            })
        output, report = module.mix(records, CharacterTokenizer())
        self.assertEqual(len(output), len(records))
        self.assertEqual(len({row["state"] for row in output}), len(records))
        self.assertEqual(sum(report["bands"].values()), len(records))
        self.assertEqual(set(report["bands"]), {str(target) for target in module.TARGETS})
        for before, after in zip(records, output):
            self.assertIn(before["state"].split("recent:\n", 1)[0], after["state"])
            self.assertTrue(after["state"].endswith(before["state"][-24:]))
            self.assertEqual(before["questions"], after["questions"])
            self.assertLessEqual(len(after["state"]) + 1, module.TARGETS[-1])
        for labels in report["labels_by_band"].values():
            self.assertIn("no_tool", labels)
            self.assertIn("Bash", labels)
        excluded = {output[0]["state"]}
        alternate, _ = module.mix(records, CharacterTokenizer(), forbidden=excluded)
        self.assertEqual(len(alternate), len(records))
        self.assertTrue(excluded.isdisjoint(row["state"] for row in alternate))

    def test_json_line_survives_kevs_splitlines_loader(self):
        encoded = module.json_line({"state": "one\u2028two\u2029three"})
        self.assertEqual(len(encoded.splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
