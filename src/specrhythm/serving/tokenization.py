"""Real Qwen3 tokenizer-only operations; no model weights, CUDA or engine startup."""

from __future__ import annotations

import json
from pathlib import Path

from specrhythm.serving.common import digest, require, sha256_file


class QwenTokenizer:
    is_real = True

    def __init__(self, path: Path):
        path = Path(path)
        require(path.is_dir(), "tokenizer directory missing", path=str(path))
        require(
            (path / "tokenizer_config.json").is_file() and (path / "tokenizer.json").is_file(),
            "Qwen3 tokenizer files missing",
        )
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError("install SpecRhythm[workload] for real CPU tokenization") from error
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path),
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        require(
            self.tokenizer.is_fast and self.tokenizer.chat_template,
            "Qwen3 requires the real fast tokenizer and chat template",
        )
        require(
            "enable_thinking" in self.tokenizer.chat_template,
            "tokenizer lacks Qwen3 enable_thinking template support",
        )
        self.identity = {
            "vocab": self.tokenizer.get_vocab(),
            "special_tokens_map": self.tokenizer.special_tokens_map,
            "special_token_ids": self.tokenizer.all_special_ids,
        }
        config = json.loads((path / "tokenizer_config.json").read_text())
        for key in ("name_or_path", "_name_or_path"):
            config.pop(key, None)
        self.fingerprint = digest(
            {
                "tokenizer_backend": json.loads(self.tokenizer.backend_tokenizer.to_str()),
                "tokenizer_config": config,
                "chat_template": self.tokenizer.chat_template,
                "identity": self.identity,
            }
        )
        self.metadata = {
            "kind": "real-qwen3-tokenizer",
            "fingerprint": self.fingerprint,
            "identity_sha256": digest(self.identity),
            "chat_template_sha256": digest(self.tokenizer.chat_template),
            "tokenizer_config_content_sha256": digest(config),
            "tokenizer_json_sha256": sha256_file(path / "tokenizer.json"),
        }

    def render(self, messages: list[dict]) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        require(isinstance(text, str) and text, "empty Qwen3 chat template result")
        return text

    def encode(self, text: str) -> list[int]:
        # The rendered template already contains special tokens.
        return list(self.tokenizer.encode(text, add_special_tokens=False))


def verify_locked_tokenizer(tokenizer, lock: dict, sources: Path) -> None:
    if lock["data_kind"] == "synthetic-fixture":
        return
    require(tokenizer.is_real, "real-data construction requires a real tokenizer")
    expected = QwenTokenizer(sources / "tokenizer")
    require(
        tokenizer.fingerprint == expected.fingerprint,
        "configured tokenizer differs from the fixed-revision source lock",
    )


def check_tokenizers(left, right, requests: list) -> dict:
    require(
        left.is_real and right.is_real, "server tokenizer verification requires real tokenizers"
    )
    require(
        left.identity == right.identity, "Draft/Target token-ID or special-token mapping differs"
    )
    for request in requests:
        for label, tokenizer in (("draft", left), ("target", right)):
            require(
                tokenizer.render(request.messages) == request.prompt_text,
                "server tokenizer template differs",
                request_id=request.request_id,
                side=label,
            )
            require(
                tokenizer.encode(request.prompt_text) == request.prompt_token_ids,
                "server tokenizer prompt encoding differs",
                request_id=request.request_id,
                side=label,
            )
    return {
        "status": "PASS",
        "validated_prompt_count": len(requests),
        "draft": left.metadata,
        "target": right.metadata,
        "model_weights_loaded": False,
        "cuda_initialized": False,
    }
