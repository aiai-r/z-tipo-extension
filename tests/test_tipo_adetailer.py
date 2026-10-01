"""Exercise the real callback bodies without loading WebUI or inference models."""

import ast
from copy import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "tipo.py"
TIMING = {"BEFORE": "before", "AFTER": "after"}


def load_callbacks():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    script = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TIPOScript")
    names = {"before_process", "process", "postprocess"}
    methods = [n for n in script.body if isinstance(n, ast.FunctionDef) and n.name in names]
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[]))
    scope = {"PROCESSING_TIMING": TIMING, "fix_seed": lambda p: None}
    exec(compile(module, str(SOURCE), "exec"), scope)
    return scope


CALLBACKS = load_callbacks()


class TIPOADetailerTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.script = SimpleNamespace(write_infotext=lambda *args: None, _process=self.infer)
        self.args = ("long", "long", "", "custom", "format", 1.0, 0.9, 40, "stub", True, False, "", "")

    def infer(self, prompt, nl_prompt, aspect_ratio, seed, *args):
        self.calls.append((prompt, seed))
        return f"expanded:{prompt}:{seed}"

    def job(self, count=10):
        return SimpleNamespace(
            prompt="input", hr_prompt="input", seed=100, width=512, height=512,
            all_prompts=["input"] * count, all_seeds=list(range(100, 100 + count)),
            enable_hr=True, all_hr_prompts=["input"] * count,
        )

    def run_callbacks(self, p, timing="AFTER", enabled=True, follow_seed=True):
        for name in ("before_process", "process"):
            CALLBACKS[name](self.script, p, enabled, TIMING[timing], 7, follow_seed, *self.args)

    def finish(self, p):
        CALLBACKS["postprocess"](self.script, p, None)

    def test_after_ten_batches_and_next_job(self):
        p = self.job()
        self.run_callbacks(p)
        expected = list(p.all_prompts)
        for _ in range(10):
            # ADetailer finishes scripts on a copy before correction, then
            # replays initialization on another copy after correction.
            self.finish(copy(p))
            inner = self.job(1)
            inner._ad_inner = True
            self.run_callbacks(inner)
            self.run_callbacks(copy(p))
        self.assertEqual(len(self.calls), 10)
        self.assertEqual([seed for _, seed in self.calls], list(range(100, 110)))
        self.assertEqual(p.all_prompts, expected)
        self.assertEqual(p.all_hr_prompts, expected)
        self.finish(p)
        self.assertFalse(hasattr(p, "_tipo_processed_owner"))
        # Also support hosts that reuse the original processing object.
        p.all_prompts = ["new input"] * 10
        self.run_callbacks(p)
        self.assertEqual(len(self.calls), 20)
        self.assertTrue(all(prompt == "new input" for prompt, _ in self.calls[10:]))
        self.run_callbacks(self.job())
        self.assertEqual(len(self.calls), 30)

    def test_before_retains_single_prompt_behavior(self):
        p = self.job()
        self.run_callbacks(p, "BEFORE")
        for _ in range(10):
            self.finish(copy(p))
            self.run_callbacks(copy(p), "BEFORE")
        self.assertEqual(self.calls, [("input", 100)])
        self.finish(p)
        p.prompt = "next"
        self.run_callbacks(p, "BEFORE")
        self.assertEqual(self.calls[-1], ("next", 100))

    def test_disabled_and_inner_jobs_do_not_infer(self):
        for timing in TIMING:
            p = self.job()
            self.run_callbacks(p, timing, enabled=False)
            p._ad_inner = True
            self.run_callbacks(p, timing)
            self.assertFalse(hasattr(p, "_tipo_processed_owner"))
        self.assertEqual(self.calls, [])

    def test_seed_offset_and_repeated_original_callback(self):
        p = self.job(6)
        self.run_callbacks(p, follow_seed=False)
        self.run_callbacks(p, follow_seed=False)
        self.assertEqual([seed for _, seed in self.calls], list(range(107, 113)))

    def test_failed_inference_does_not_mark_job_complete(self):
        p = self.job()
        self.script._process = lambda *args: 1 / 0
        with self.assertRaises(ZeroDivisionError):
            self.run_callbacks(p)
        self.assertFalse(hasattr(p, "_tipo_processed_owner"))
        self.script._process = self.infer
        self.run_callbacks(p)
        self.assertEqual(len(self.calls), 10)


if __name__ == "__main__":
    unittest.main()
