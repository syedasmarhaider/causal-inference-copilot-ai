# Supplementary Note 1 — Estimator Knowledge-Library Construction

The estimator selector uses a fixed EconML registry and a curated estimator knowledge library stored in repository text files under `src/python/implementation/workflows/tools/causal/inference/econml/info_txt/`. The selector cannot introduce arbitrary estimator classes; recommendations must map to one of the supported registry keys.

The source documents recorded in the library are primarily EconML documentation pages, with DR learner files also recording the EconML DR learner source URL. Each curated file summarizes estimator purpose, identifying assumptions, supported inputs, inference behavior, practical limitations, and compatibility notes. The fixed registry and adapter code are the final source of truth for which estimators can actually run.

The current repository does not store a machine-readable provenance record for an external GPT-5.5 Thinking summarization step or access date. If external GPT-assisted summarization is used for the manuscript, the access date, prompt/version, source URLs, reviewer, and source-check outcome should be recorded separately. In this repository snapshot, the committed curated text files and their retained source URLs are the auditable artifacts.

Intermediate summaries should not be treated as authoritative causal or software documentation. They are selector context only. The implementation still enforces estimator support through `CausalModelFactoryTool`, adapter imports, typed model keys, validation gates, and training-time errors from static Python code.
