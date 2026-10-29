import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name("cccp_tool_decisions.py")
SPEC = importlib.util.spec_from_file_location("cccp_tool_decisions", MODULE)
DECISIONS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DECISIONS)

def claude_rows():
    return [
        {"type": "user", "message": {"content": "Find the failing test"}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "I will inspect it"}, {"type": "tool_use", "name": "Read", "input": {"file_path": "test.py"}}], "stop_reason": "tool_use"}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "content": "the secret result"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "The test fails on line 2"}], "stop_reason": "end_turn"}},
    ]

def codex_rows():
    return [
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Find the failing test"}]}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{\"cmd\":\"cat test.py\"}"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "output": "the secret result"}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "phase": "final_answer", "content": [{"type": "output_text", "text": "Found it"}]}},
    ]

class DecisionsTest(unittest.TestCase):
    def test_decision_states_carry_exactly_the_prior_context(self):
        for events, label in [(DECISIONS.claude_events, "Read"), (DECISIONS.codex_events, "exec_command")]:
            history = []
            snapshots = []
            for kind, value in events(claude_rows() if label == "Read" else codex_rows()):
                if kind == "decision":
                    snapshots.append(("\n".join(history), value))
                else:
                    history.append(value)
            self.assertEqual([label, "no_tool"], [item[1] for item in snapshots])
            self.assertNotIn("secret result", snapshots[0][0])
            self.assertIn("secret result", snapshots[1][0])
        nested = [
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Search the repo"}]}},
            {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec", "input": "const r=await tools.exec_command({cmd:'rg foo'}); text(r)"}},
        ]
        self.assertEqual(["exec_command"], [value for kind, value in DECISIONS.codex_events(nested) if kind == "decision"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "mirror-codex" / "sessions"
            root.mkdir(parents=True)
            rows = [
                {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Investigate " + "x" * 3000}]}},
                {"type": "response_item", "payload": {"type": "function_call_output", "output": "First result " + "a" * 3000}},
                {"type": "response_item", "payload": {"type": "function_call_output", "output": "Second result " + "b" * 3000}},
                {"type": "response_item", "payload": {"type": "function_call", "name": "exec_command", "arguments": "{}"}},
            ]
            (root / "long.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            _, _, pairs = next(DECISIONS.examples([root]))
            state, label = pairs[0]
            self.assertEqual("exec_command", label)
            self.assertGreater(len(state), 6000)
            self.assertIn("First result", state)
            self.assertIn("Second result", state)

    def test_dataset_is_session_split_and_has_two_questions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            claude = root / "mirror-claude" / "projects"
            codex = root / "mirror-codex" / "sessions"
            claude.mkdir(parents=True)
            codex.mkdir(parents=True)
            for index in range(12):
                target = (claude if index % 2 == 0 else codex) / f"session-{index}.jsonl"
                rows = claude_rows() if index % 2 == 0 else codex_rows()
                if index % 2 == 0:
                    rows[0]["message"]["content"] = f"Find test {index}"
                else:
                    rows[0]["payload"]["content"][0]["text"] = f"Find test {index}"
                target.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (claude / "duplicate.jsonl").write_text((claude / "session-0.jsonl").read_text())
            out = root / "output"
            counts = DECISIONS.dataset([claude, codex], out, 50)
            self.assertEqual(sum(counts.values()), 24)
            self.assertEqual(26, sum(json.loads((out / "manifest.json").read_text())["raw_counts"].values()))
            rows = [json.loads(line) for name in ("train", "eval") for line in (out / f"{name}.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), len({row["state"] for row in rows}))
            self.assertEqual({"call_tool", "tool"}, set(rows[0]["questions"]))
            self.assertIn("latest user: Find test", rows[0]["state"])
            self.assertEqual({"no_tool", "Read"}, set(next(row for row in rows if row["state"].startswith("agent: claude"))["questions"]["tool"]["criteria"]))
            self.assertEqual({"no_tool", "exec_command"}, set(next(row for row in rows if row["state"].startswith("agent: codex"))["questions"]["tool"]["criteria"]))
            self.assertTrue((out / "manifest.json").exists())
            kev_out = root / "kev"
            limited = DECISIONS.dataset([claude, codex], kev_out, 50, "kev", 5)
            self.assertEqual(5, limited["train"])
            kev_rows = [json.loads(line) for line in (kev_out / "train.jsonl").read_text().splitlines()]
            self.assertEqual({"tool"}, set(kev_rows[0]["questions"]))
            self.assertIn("label", kev_rows[0]["questions"]["tool"])
            self.assertEqual(counts["train"], json.loads((kev_out / "manifest.json").read_text())["train_available"])

    def test_prepared_dataset_uses_existing_files_on_gpu_host(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prepared = root / "prepared"
            prepared.mkdir()
            for name in ("train.jsonl", "eval.jsonl", "manifest.json"):
                (prepared / name).write_text("{}\n")
            run = root / "run"
            argv = ["cccp_tool_decisions.py", "--prepared", str(prepared), "--base", str(root / "base"), "--out", str(run)]
            with patch("sys.argv", argv), patch.object(DECISIONS.subprocess, "run") as invoked:
                DECISIONS.main()
            command = invoked.call_args.args[0]
            self.assertEqual(str(prepared / "train.jsonl"), command[command.index("--data") + 1])
            self.assertEqual(str(prepared / "eval.jsonl"), command[command.index("--eval-data") + 1])
            self.assertTrue((run / "training.log").exists())

if __name__ == "__main__":
    unittest.main()
