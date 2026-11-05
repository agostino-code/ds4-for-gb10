# Agent Notes

`ds4.c` is a DeepSeek V4 Flash specific inference engine. It is not a generic
GGUF runner. The goal is a small, readable, high-performance C/CUDA codebase
specialized for the NVIDIA DGX Spark (GB10 / SM121a). Metal and ROCm backends
are stripped; CUDA is the only GPU backend.

## Goals

- Keep the production path as whole-model CUDA graph inference tuned for
  `sm_121a` (D2R MoE, FP4 index, Blackwell fast-path kernels).
- Always make sure that the SSD streaming, CUDA, distributed inference, and
  CPU reference paths are not affected by fixes to other parts of the code.
- Keep model loading mmap-backed for the CUDA resident case; do not eagerly
  copy the full GGUF. Keep the model loading for SSD streaming of routed
  experts explicit: allocated buffers, fast reads from disk, always try to
  hide loading of missing routed experts by loading them while performing the
  inference of the shared expert and routed experts already in RAM. Always try
  to hide loading of layers for prefill in SSD streaming mode using the
  inference time of the current layer as the next one is loaded.
- Keep the CPU backend CPU-only and use it only as reference/debug code.
- Preserve correctness before speed. Do not keep a faster path with unexplained
  attention, KV cache, or logits drift.
- Make long local agent sessions practical through live KV reuse and disk KV
  checkpoints.

## Quality Rules

- Keep the implementation small, sharp, easy to understand. Try to write
  elegant code in a state of grace. Don't settle for the first thing that comes
  to mind, try to find the most minimal and better working design. Don't
  introduce slop: very fragile code that just patches specific cases, dead
  code, useless code and code ways more complicated of how it should be.
- Comment important inference code where the model mechanics, cache lifetime,
  memory policy, or API orchestration are not obvious from the local code.
- Prefer comments beside the implementation over separate design documents.
- Keep comments instructive and compact: explain why a shape, ordering, cache
  boundary, or memory choice exists.
- Keep public APIs narrow. CLI/server code should not know tensor internals.
- Do not add permanent semantic variants behind flags. Diagnostic switches are
  fine when they validate the one release path.
- Do not introduce C++.

## Safety

- Avoid large CPU inference runs; the CPU path is a reference/debug backend and
  has previously exposed kernel VM failures with very large mappings.
- Do not run multiple huge model processes concurrently. The instance lock is
  intentional.

## Layout

- `ds4.c`: model loading, tokenizer, CPU reference code, CUDA graph scheduling,
  sessions, disk-cache payload serialization.
- `ds4_cli.c`: command line, linenoise REPL, interactive transcript handling.
- `ds4_server.c`: OpenAI/Anthropic compatible HTTP API, worker queue, streaming,
  tool-call mapping, disk KV cache policy.
- `ds4_cuda.cu`: CUDA kernels and Blackwell `sm_121a` fast paths.
- `cuda/mmq/`: vendored llama.cpp mmq prefill tier (D2R MoE tensor cores).
- `tests/`: unit and live integration tests.

This list is not complete, check the files for more info.

## Testing

Use `make` for build validation. Use `make test` for unit/regression tests when
a model and CUDA are available. Use live server tests only when intentionally
testing the API surface.

At every major change where one of the following could be affected, make sure
to:

1. Test the normal CUDA path and that speed is still at the level it was.
2. Test the SSD streaming path.
3. Test the distributed inference if it could be affected, but ask the user
   before doing so.
4. Check if CUDA is broken after the change, and ask the user to give you
   access to the CUDA machine to actually test if everything is still fine.
