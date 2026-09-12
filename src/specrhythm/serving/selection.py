"""Auditable real-text extraction and deterministic, globally deduplicated selection."""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from specrhythm.serving.common import (
    ALGORITHM,
    TASKS,
    canonical,
    digest,
    normalized_prompt,
    require,
    source_path,
    text_hash,
)
from specrhythm.serving.sources import records

COLUMNS = {
    "chat": ["message_id", "message_tree_id", "text", "role", "lang", "parent_id", "deleted"],
    # Official APPS JSONL calls the ID 'id'; its old loader renames it to problem_id.
    "code": ["id", "question", "starter_code", "difficulty", "url"],
    "summarization": ["id", "article"],
    "reasoning": ["question"],
}
CHAT_RULES = {
    "chat_code_task": r"```|\b(code|coding|programming|python|javascript|typescript|c\+\+|"
    r"algorithm|debug|sql|write a function|write a program)\b",
    "chat_math_task": r"\b(solve|calculate|equation|algebra|integral|derivative|arithmetic|"
    r"math|mathematics|prove|probability)\b|\b(?:[a-z]|\d+)\s*[+*/=]\s*(?:[a-z]|\d+)\b",
    "chat_summary_task": r"\b(summarize|summarise|summarization|summarisation|"
    r"summary|tldr)\b|tl;dr",
}
COMPILED_CHAT_RULES = [
    (name, re.compile(pattern, re.IGNORECASE)) for name, pattern in CHAT_RULES.items()
]


def extract(task: str, row: dict, config: dict):
    """Only whitelisted input fields contribute; never answers, highlights or tests."""
    if task == "chat":
        for reason, keep in (
            ("chat_non_prompter", row["role"] == "prompter"),
            ("chat_non_english", row["lang"] == "en"),
            ("chat_non_root", row["parent_id"] is None),
            ("chat_deleted", row["deleted"] is False),
        ):
            if not keep:
                return None, reason
        text, record_id, group = row["text"], row["message_id"], row["message_tree_id"]
        require(isinstance(text, str), "chat text is not a string")
        for reason, pattern in COMPILED_CHAT_RULES:
            if pattern.search(text):
                return None, reason
    elif task == "code":
        require(
            row["difficulty"] in ("introductory", "interview", "competition"),
            "unknown APPS difficulty in all configuration",
        )
        text, record_id, group = row["question"], row["id"], row["url"] or None
        starter = row["starter_code"]
        require(isinstance(text, str) and isinstance(starter, str), "invalid APPS input fields")
        if starter.strip():
            text += "\n\nStarter code:\n" + starter
    elif task == "summarization":
        text, record_id, group = row["article"], row["id"], row["id"]
    else:
        require(task == "reasoning", "unknown source adapter")
        text = row["question"]
        require(isinstance(text, str), "GSM8K question is not a string")
        # GSM8K main has no native ID. Exact question content is the stable identity.
        record_id = "question-sha256:" + text_hash(normalized_prompt(text))
        group = record_id
    require(isinstance(text, str), "source prompt field is not text", task=task)
    if not text.strip():
        return None, "empty_input"
    require(type(record_id) in (str, int) and str(record_id), "missing source record ID")
    require(group is None or isinstance(group, str), "invalid source group ID")
    return {
        "content": config["instructions"][task] + text,
        "id": str(record_id),
        "group_id": group,
    }, None


def choose(lock: dict, root: Path, config: dict, tokenizer) -> tuple[dict, dict]:
    """Disk spool bounds RAM on the full CNN/DM train pool. No model scoring."""
    report = {
        "algorithm": ALGORITHM,
        "data_kind": lock["data_kind"],
        "chat_filter_rules": CHAT_RULES,
        "classification_scope": "dataset provenance and fixed rules; not perfect semantic labels",
        "length_filter_scope": "ranked unique candidates examined until both split quotas fill",
        "by_class": {
            task: {
                "source_rows": 0,
                "input_eligible": 0,
                "unique_candidates": 0,
                "length_checked": 0,
                "unexamined_after_quota": 0,
                "filters": Counter(),
                "duplicates": Counter(),
                "selected": {"main": 0, "calibration": 0},
            }
            for task in TASKS
        },
    }
    selected = {split: {task: [] for task in TASKS} for split in ("main", "calibration")}
    with tempfile.TemporaryDirectory(prefix="specrhythm-s0-candidates-") as temporary:
        connection = sqlite3.connect(str(Path(temporary) / "candidates.sqlite"))
        try:
            connection.execute(
                "CREATE TABLE candidates (rank TEXT, task TEXT, ref TEXT, "
                "content TEXT, prompt_hash TEXT)"
            )
            for task in TASKS:
                source = lock["sources"][task]
                stats = report["by_class"][task]
                for file in sorted(source["files"], key=lambda f: f["path"]):
                    if file["role"] != "data":
                        continue
                    for index, row in records(
                        source_path(root, task, file["path"]), COLUMNS[task]
                    ):
                        stats["source_rows"] += 1
                        prompt, reason = extract(task, row, config)
                        if reason:
                            stats["filters"][reason] += 1
                            continue
                        stats["input_eligible"] += 1
                        ref = {
                            "dataset_id": source["dataset_id"],
                            "revision": source["revision"],
                            "config": source["config"],
                            "split": source["split"],
                            "id": prompt["id"],
                            "group_id": prompt["group_id"],
                            "file": file["path"],
                            "file_sha256": file["sha256"],
                            "original_row_index": index,
                        }
                        text = tokenizer.render([{"role": "user", "content": prompt["content"]}])
                        rank = digest(
                            [
                                "candidate",
                                config["seed"],
                                source["dataset_id"],
                                prompt["id"],
                                text_hash(text),
                            ]
                        )
                        connection.execute(
                            "INSERT INTO candidates VALUES (?,?,?,?,?)",
                            (
                                rank,
                                task,
                                canonical(ref).decode(),
                                prompt["content"],
                                text_hash(normalized_prompt(text)),
                            ),
                        )
            connection.commit()
            seen_ids, seen_groups, seen_prompts = set(), set(), set()
            for _, task, raw_ref, content, prompt_hash in connection.execute(
                "SELECT * FROM candidates ORDER BY rank, ref"
            ):
                stats, ref = report["by_class"][task], json.loads(raw_ref)
                sid = (ref["dataset_id"], ref["id"])
                group = (ref["dataset_id"], ref["group_id"]) if ref["group_id"] else None
                reason = (
                    "source_id"
                    if sid in seen_ids
                    else "source_group"
                    if group and group in seen_groups
                    else "normalized_prompt"
                    if prompt_hash in seen_prompts
                    else None
                )
                if reason:
                    stats["duplicates"][reason] += 1
                    continue
                seen_ids.add(sid)
                if group:
                    seen_groups.add(group)
                seen_prompts.add(prompt_hash)
                stats["unique_candidates"] += 1
                split = next(
                    (
                        s
                        for s in ("main", "calibration")
                        if len(selected[s][task]) < config["splits"][s]["quotas"][task]
                    ),
                    None,
                )
                if split is None:
                    stats["unexamined_after_quota"] += 1
                    continue
                messages = [{"role": "user", "content": content}]
                text = tokenizer.render(messages)
                ids = tokenizer.encode(text)
                stats["length_checked"] += 1
                limits = config["class_limits"][task]
                if len(ids) > limits["prompt_tokens"]:
                    stats["filters"]["prompt_length_limit"] += 1
                    continue
                if (
                    len(ids) + limits["maximum_new_tokens"] + config["speculative_reserve_tokens"]
                    > config["max_model_len"]
                ):
                    stats["filters"]["context_budget"] += 1
                    continue
                require(bool(ids), "tokenizer produced an empty prompt")
                selected[split][task].append(
                    {
                        "source_ref": ref,
                        "messages": messages,
                        "prompt_text": text,
                        "prompt_token_ids": ids,
                    }
                )
                stats["selected"][split] += 1
        finally:
            connection.close()
    for task in TASKS:
        for split in selected:
            require(
                len(selected[split][task]) == config["splits"][split]["quotas"][task],
                "candidate pool exhausted before exact quotas",
                task=task,
                split=split,
                actual=len(selected[split][task]),
                required=config["splits"][split]["quotas"][task],
                selection_report=report,
            )
    return selected, report
