# Data directory

Place downloaded source traces under data/raw/ and normalized or generated workloads under
data/processed/. Both directories are intentionally ignored by Git. Do not commit raw prompts,
user content, model outputs, or large third-party traces.

The repository commits only schemas, small fixtures, configuration files, and provenance
documentation. Keep generated JSONL, validation reports, and manifests in the external data tree;
the manifest contains portable filenames/relative paths plus content hashes, so an absolute local
path is never the only provenance.

`configs/workloads/r3-mooncake-622-proxy.json` combines Mooncake arrival timestamps with sampled
proxy token lengths. It does not contain HumanEval, Alpaca, or CNN/DailyMail examples and is not the
final R3 evaluation dataset. `specrhythm import-mooncake` is the separate R4 path: it retains
Mooncake timestamp, length, and prefix-hash fields. Neither path provides a measured speculative
acceptance profile; collect that on the GPU server for the exact model pair before reporting system
performance.

Recommended external layout on the GPU server:

~~~text
/home/rzwang/data/SpecRhythm-data/
├── raw/Mooncake/FAST25-release/traces/conversation_trace.jsonl
├── processed/workload-v0.1/r3-proxy/
└── manifests/workload-v0.1/
~~~

See `docs/workload-design.md` for the complete generate, manifest, validate, and summarize command
sequence using these paths.
