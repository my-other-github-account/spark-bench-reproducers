# License and privacy review

## Release decision

Status: PASS WITH DECLARED EXTERNAL DEPENDENCIES

This bundle is a research evidence recovery package, not a model or third-party software distribution. Every shipped payload is UTF-8 text. The release excludes model weights, checkpoints, tensor files, compiled binaries, caches, package archives, container layers, and symlinks.

## Content classes

### First-party mission code and receipts

Classification: included.

The recovered runners, adapters, kernels, launchers, finalizers, manifests, logs, and receipts were produced inside the project mission/board authority chain. Each file has an exact source SHA-256 and a shipped SHA-256 in `SOURCE_MANIFEST.json`. Runtime-specific paths, direct addresses, and operator-local identifiers were deterministically replaced with placeholders.

### Generated benchmark outputs

Classification: included for reproducibility and receipt audit.

Selected HumanEval/EvalPlus raw and sanitized completions are included only where they are directly pinned by P486, P234, or P968 receipts. They contain generated candidate code and benchmark task identifiers, not model parameters. Their purpose is to allow sanitizer/row-count/hash verification. Automated privacy scanning is mandatory before release.

### EvalPlus and benchmark dependencies

Classification: not redistributed.

The bundle includes first-party wrappers and pin receipts but not the EvalPlus source tree, Docker image filesystem, or benchmark package distribution. Consumers must obtain the external dependency under its own license and verify the exact commit, image identity, dataset name, and dataset SHA-256 recorded by the receipts.

### Models, tensors, packed artifacts, and runtimes

Classification: excluded.

No `.pt`, `.pth`, `.safetensors`, `.gguf`, `.bin`, `.onnx`, NumPy tensor archive, model cache, CUDA binary, or container archive is shipped. Hashes that appear in receipts are identifiers only.

## Privacy controls

The recovery builder performs deterministic substitution of:

- operator home directories;
- collection-host home directories;
- scratch and temporary roots;
- direct LAN/QSFP/Tailscale addresses;
- operator-local username and hostname literals.

The release scanner checks all shipped text except its own regex source for:

- common cloud/Hugging Face/API credential forms;
- high-risk secret assignments;
- private absolute home/runtime paths;
- direct cluster addresses;
- email addresses and MAC addresses;
- operator-local identifiers;
- forbidden model/tensor file extensions;
- symlinks and non-regular filesystem entries.

The scanner never prints a suspected secret value; it emits only rule IDs, relative paths, and line numbers. Any finding fails the release.

## File-by-file traceability

`SOURCE_MANIFEST.json` carries `license_classification` and `privacy_review` for every recovered source. `BUNDLE_MANIFEST.json` inventories every final shipped file. `MISSING_SOURCE_LEDGER.json` records omitted external dependencies and the effect of each gap.

## Consumer obligations

1. Run both bundled verification tools after extraction.
2. Rebind placeholders only in a task-local copy.
3. Resolve third-party dependencies under their own licenses.
4. Do not publish generated benchmark outputs as original human-authored library code.
5. Do not infer model availability from a receipt hash.
6. Do not promote P948, P950, or P526 beyond the explicit dispositions in `ADOPTION_MAP.json`.
