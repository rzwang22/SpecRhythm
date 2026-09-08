# Phase S0: mixed real-text workload foundation

Phase S means **Serving Workloads & Arrival Replay**. S0 constructs and validates
data. Phase 4C continues to mean Dual-Eager. S0 does not rename it or change the
resident decode-only Phase 4 experiments. Arrival submission, dynamic mixed
prefill/decode, Serial/PingPong admission, SLO calibration and PD/KV handoff are
future work. There is no inference or GPU dependency in this pipeline.

## Frozen recipe

The versioned recipe is `configs/workloads/mixed-real-v1.json`. The description is
**mixed-task real-text workload with Mooncake arrival replay**. These project
quotas are not an estimate of Mooncake's production task distribution.

| Task | Main | Calibration | Templated prompt limit | Maximum new tokens |
| --- | ---: | ---: | ---: | ---: |
| chat | 300 | 50 | 1024 | 512 |
| code | 300 | 50 | 1536 | 1024 |
| summarization | 200 | 50 | 3072 | 512 |
| reasoning | 200 | 50 | 1024 | 1024 |

The main/calibration IDs are `mixed-real-v1-main1000` and
`mixed-real-v1-calibration200`. Selection seed is 1664; the independent class-slot
arrival seed is 1665. Every request is English, one user message, rendered using
the actual Qwen3 template with `add_generation_prompt=true` and
`enable_thinking=false`. Encoding uses `add_special_tokens=False`: the rendered
template already contains special tokens. Instructions are versioned in the
recipe, including the fixed summary and mathematics instructions.

The prompt length is the actual token-ID count. It must satisfy the class limit
and `prompt_length + maximum_new_tokens + 4 <= 4096`. Overlong candidates are
filtered and replaced in deterministic order; input text is never truncated or
padded. This is a conservative data constraint, not a physical KV allocation
formula. Natural EOS remains enabled. `maximum_new_tokens` includes the first
generated token and is an upper bound, never a measured output length. Bootstrap
measurement boundaries belong to future runners.

## Audited sources and acquisition

The committed `mixed-real-v1-source-lock.json` contains exact file URLs, revisions,
byte SHA256 values, sizes and attribution. Each original train file is retained;
no conversion or dataset loading script is executed. The lock covers 23 files,
including dataset cards, trace documentation/license and tokenizer configuration.

| Input | Immutable revision | Official data files / extraction |
| --- | --- | --- |
| [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | One `data/train-*.parquet`; English, undeleted `prompter` roots; `message_id` / `message_tree_id` |
| [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps) | `21e74ddf8de1a21436da12e3e653065c5213e9d1` | `train.jsonl`, all three difficulties; question and original starter code |
| [abisee/cnn_dailymail](https://huggingface.co/datasets/abisee/cnn_dailymail) | `96df5e686bee6baa90b8bee7c28b81fa3fa6223d` | Three `3.0.0/train-*.parquet`; article and native article ID |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f781978c0658806c59bc2815b9866` | `main/train-*.parquet`; question; stable question-content SHA because no native ID exists |
| [Mooncake FAST25 trace](https://github.com/kvcache-ai/Mooncake/tree/2ca843e0073c4acd3c2beb224f82ff4df687c395/FAST25-release/traces) | `2ca843e0073c4acd3c2beb224f82ff4df687c395` | Entire `conversation_trace.jsonl`; timestamp only |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca) | `c1899de289a04d12100db370d81485cdf75e47ca` | Explicit tokenizer/config/card/license allowlist; no model weights |

The APPS raw JSONL field is **`id`**; the historical `apps.py` script renames it to
`problem_id`. The adapter reads the raw field. APPS `solutions` and hidden
`input_output`, CNN/DM `highlights`, and GSM8K `answer` cannot contribute to prompts.
Parquet uses projected input columns; JSONL parsing immediately projects the
allowed fields. Original examples inside a programming question remain intact.

Dataset cards declare Apache-2.0 for OASST1/CNN-DM and MIT for APPS/GSM8K. The lock
records these source declarations, rather than making an independent legal claim;
the retained cards provide citation/attribution details. Mooncake's actual license
filename is `LICENSE-APACHE`. Source resolve is an explicit one-time operation;
ordinary fetch/build use the committed immutable lock and do not resolve `main`.

Acquisition reuses the chosen Hugging Face cache (`<cache>/hub`), supports a local
import map and an offline mode, and links verified original files into a logical
sources directory. Missing fields, changed hashes, incomplete train-file layouts
and candidate exhaustion fail closed. No alternate source or synthetic fallback
is available in the CLI. Source cards are data and are never executed.

## Selection and independent split isolation

`sha256-rank-global-dedup-then-quota-v1` renders candidates from the allowed source
fields and ranks them by SHA256 of canonical JSON containing the seed, dataset,
stable source ID and full prompt hash. Canonical source references break ties.
A temporary SQLite spool bounds memory on the full CNN/DM pool; filesystem order,
Python randomized hashes and download order do not affect selection.

Candidates from all four classes enter one ranked pool. Before assignment, the
algorithm deduplicates dataset/sample ID, dataset/group ID when present, and the
normalized full prompt hash across that pool. Normalization changes only CRLF to
LF; spaces, code indentation and Unicode remain intact. Unique candidates fill
main then calibration quotas per class. Failed length checks consume no quota.
Unselected unique candidates are still counted for deduplication, but are not
tokenized after the two quotas fill. The reported length-filter fraction is
therefore explicitly conditional on examined candidates, not a full-corpus
length census. Exhaustion reports actual counts and filter/dedup accounting.

Chat applies ordered, case-insensitive regex rules for code, mathematics and
summary requests after role/language/root/deletion checks. Exact patterns and
first-matching exclusion counts are in `selection-report.json` and
`serving/selection.py`; mathematical expressions include single-letter variables
such as `3*x` and `X+Y`. Classification is source/rule based, not perfect semantic
labeling. No Target output, acceptance, performance or model scoring affects
selection. `sample-review.md` contains the first five main prompts in arrival
order for each class, with source references and token counts.

## Arrival and request identity

Every trace row is read and its timestamp validated, including rows outside the
selected windows. Sort order is `(timestamp, original_line_index)`; line indices
are **one-based**, sorted indices and dataset row indices are **zero-based**.
Main takes sorted positions `[0,1000)`, calibration `[1000,1200)`. Each split
subtracts its own first timestamp and retains all ties and gaps, in milliseconds.
SHA-ranked class slots using seed 1665 assign exact quotas to these unchanged time
positions. Window origins, original line IDs, trace revision and SHA are retained.

The future mapping is `scheduled_arrival_time_ms = arrival_time_ms / time_scale`,
where the scale is finite and positive and the result must remain finite. S0
does not submit requests or sweep load. Trace lengths and prefix hashes are
discarded: they provide neither prompt/output lengths nor KV identity, and no
original production joint distribution is reconstructed.

Request IDs hash workload family, task, source identity/revision, full prompt and
tokenizer identity. Sampling seeds hash selection seed and request ID. Changing
arrival seed/time scale does not change payload identity. The independent
`ServingWorkloadRequest` loader accepts any positive legal request count; exact
1000/200 quotas are a recipe/validator requirement, not a legacy runner constraint.

Input schema includes source, messages, templated text, tokens, budgets, stable
identities and arrivals. No generated-output or performance result fields exist.
SLO class equals task class. `slo_policy_ref=null` and `calibration_status=pending`
cannot pass `require_calibrated`; later calibration belongs in a separate
experiment configuration associated with the workload hash. Old R3 SLO values
are not copied.

## Tokenizer and artifact identities

The tokenizer fingerprint hashes backend tokenizer JSON, path-independent
tokenizer configuration, actual vocabulary/token-ID mapping, special-token map
and IDs, and chat template. The configured real tokenizer must match the locked
tokenizer before build/validation. Runtime package versions belong to build-info.
Server `tokenizer-check` additionally compares the two real model-directory
tokenizers and renders/encodes every formal prompt identically. Preflight without
prompts is only `READY`, with prompt acceptance still `PENDING`.

Canonical JSON is UTF-8, sorted keys, compact separators, no nonfinite numbers,
and one trailing newline. The semantic workload SHA256 hashes the core manifest
before adding its own hash field. Config/source-lock content hashes use that same
canonical representation. `prompt_sha256` hashes exact prompt UTF-8 bytes.
File checksums hash actual file bytes; a manifest's file checksum is distinct
from its semantic workload hash. Neither contains wall-clock time or local paths.

The manifest binds source/file hashes, recipe, seeds, quotas, tokenizer identity,
instruction/template/context contract, arrival windows and both JSONL hashes.
`build-info.json` separately records host, Python/dependency versions, paths,
elapsed time and execution commit/dirty state. Two builds can have different
build-info/logs while yielding byte-identical JSONL and core manifest.

The validator rereads all JSONL rows, sources and the entire trace. It checks
counts, quotas, all three dedup identities, cross-split isolation, exact template
and tokens, context budgets, source extraction, arrival mapping/order, identities,
and hashes. It independently re-extracts selected raw records and recomputes the
full deterministic selection and derived reports. It never trusts a builder
`valid=true`, truncates its read at a quota, or repairs inputs. Source/artifact
hashes are compared before and after validation. Errors are structured and cause
a nonzero CLI return. `seal` checks that validated artifacts remain unchanged,
then inventories all top-level artifact/log files, excluding `checksums.sha256`
itself. The review tar contains these small files, never the original datasets.

## Implementation boundary and evidence

New modules live under `src/specrhythm/serving/`: schema, sources, tokenization,
selection, arrival, builder, validation and CLI. SHA file hashing reuses the pure
Phase 3 utility. Older arrival helpers discard original line identities and/or
rebase the combined window, so S0 keeps a separate timestamp composer. Legacy
workload/SmokeRequest/CLI/runners remain unchanged. Heavy data dependencies are
lazy imports in the optional `workload` extra; fixture CI requires only `dev`.

The local Python 3.11 real-data build uses the pinned remote tokenizer. It produced
1000/200 requests with full validator PASS and no input mutation. Main prompt
length p50/p90/max are chat 28/65/319, code 370/732/1329, summary 820/1523/2262,
reasoning 78/110/235. Main coverage is 330000 ms (889 equal adjacent pairs);
calibration coverage is 80999 ms (172 pairs). The summary reports nearest-rank
percentiles and `(N-1)/span` arrival rate; singleton/zero-span rates are null.

| Final local artifact | SHA256 |
| --- | --- |
| Semantic workload | `05b5f2efad0a4ac1771c9a18d847c08d95d360432e1c9cfa55a82ba2f63cba02` |
| `main1000.jsonl` bytes | `19979cc335b48a2a5c2d64e2a18c29b0b523d9a970733f4c6744eb5cea4779ef` |
| `calibration200.jsonl` bytes | `7a1278243b7d2aa5a71d4082eafb22f29dfa5e281b047c1443295684eb7652b9` |
| `workload-manifest.json` bytes | `a2cb9bda2ad723cf6ac1050bca7e0a7818574d6617b9352bc30db70b0cce9703` |

The final source scan covers 84437 OASST1, 5000 APPS, 287113 CNN/DM and 7473 GSM8K
train rows. Chat has 3038 input-eligible rows and 25 duplicate prompts; CNN/DM has
3108 duplicate prompts. APPS examines 352 unique candidates to select 350,
filtering two for prompt length. Detailed first-reason counts are retained.

Local real-data PASS is distinct from synthetic fixture PASS and from
**server-local tokenizer verification PENDING**. Local builds record the starting
implementation commit and dirty state honestly; no server acceptance is inferred.
Follow the [CPU runbook](phase-s0-workload-runbook.md) for full server data
acceptance and sealing. Stop for data review after S0; S1–S5 remain future work.
