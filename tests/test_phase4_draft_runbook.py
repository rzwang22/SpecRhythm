"""Check operator commands without executing any GPU command or connecting remotely."""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_phase4_vllm_draft import next_token

from specrhythm.phase4 import draft_gate


def test_runbook_shell_and_embedded_python_syntax():
    text = Path("docs/phase4b3-plan-a-runbook.md").read_text()
    blocks = re.findall(r"```bash\n(.*?)```", text, re.S)
    assert len(blocks) >= 20
    assert blocks[0].startswith("set +e\nset +u\nset +o pipefail\ntrap - ERR\n")
    for block in blocks:
        assert not re.search(r"\bexit\b|set -e", block)
        assert 'RC="$?"' in block and "rc=$RC" in block
        result = subprocess.run(["bash", "-n"], input=block, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        for python in re.findall(r"<<'PY'\n(.*?)\nPY\n", block, re.S):
            ast.parse(python)


def test_gpu_gate_requires_explicit_opt_in_before_loading_config(monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["gate", "--config", "absent", "--gate", "D1", "--output", "absent"]
    )
    with pytest.raises(SystemExit) as caught:
        draft_gate.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("count", [1, 2, 4, 8])
def test_oracle_fixture_covers_multiround_budget_and_eos_without_gpu(monkeypatch, tmp_path, count):
    class Oracle:
        def __init__(self, config):
            self.states, self.base, self.pending = {}, {}, {}
            self.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))

        def initialize(self, rid, prefix):
            self.states[rid] = tuple(prefix)

        def propose(self, rid, budget, eos):
            prefix = self.states[rid]
            self.base[rid] = prefix
            tokens = []
            for _ in range(budget):
                token = next_token(prefix + tuple(tokens))
                tokens.append(token)
                if token in eos:
                    break
            self.pending[rid] = tuple(tokens)
            return tuple(tokens), max(len(tokens) - 1, 0)

        def rollback(self, rid, accepted):
            self.states[rid] = self.base[rid] + self.pending[rid][:accepted]

        def append_target_token(self, rid, token):
            self.states[rid] += (token,)

        def finish(self, rid):
            self.states.pop(rid)

        def shutdown(self):
            assert self.states == {}

    monkeypatch.setattr(draft_gate, "HFPersistentDraftBackend", Oracle)
    (tmp_path / "config.json").write_text('{"vocab_size": 100}')
    config = SimpleNamespace(draft=SimpleNamespace(resolved_model_path=tmp_path))
    tokenizer = SimpleNamespace(eos_token_id=None, encode=lambda text: [ord(c) % 79 for c in text])
    fixtures = draft_gate._hf_fixture(
        config, [("normal", count, False), ("eos", count, True)], tokenizer
    )
    normal, eos = fixtures
    assert len(normal["rounds"]) == 4
    for i, rid in enumerate(normal["initial"]):
        generated = sum(
            len(
                next(r for r in item["synchronizations"] if r["request_id"] == rid)[
                    "committed_delta"
                ]
            )
            for item in normal["rounds"]
        )
        assert generated == 10 + i % 3
    assert eos["rounds"][0]["synchronizations"][0]["terminal"] is True
    assert all(
        row["request_id"] != "eos-0" for item in eos["rounds"][1:] for row in item["proposals"]
    )
