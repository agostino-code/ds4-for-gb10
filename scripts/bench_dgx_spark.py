#!/usr/bin/env python3
"""
bench_dgx_spark.py - DGX Spark (GB10 / SM121) performance benchmark suite.

Python port of scripts/bench_dgx_spark.sh. Assumes the environment is already
at the tuned "sweet spot": model + DSpark drafter downloaded, `make cuda-spark`
built, and Blackwell fast-path env vars are set here exactly as the shell script
does. No extra setup or context-testing logic is included.

Runs three ds4-bench sweeps and prints a per-frontier summary:
  1. Pure prefill sweep (D2R MoE tensor-core throughput)
  2. Plain single-token decode (LPDDR5X bandwidth roofline)
  3. DSpark speculative decode (if the drafter is present)

Usage:
  python3 scripts/bench_dgx_spark.py [--prompt-file FILE] [--model FILE] [--drafter FILE]
"""

import os
import subprocess
import sys
import csv
import io

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "ds4-bench")

# Candidate model / drafter paths, same as bench_dgx_spark.sh.
MODEL_CANDIDATES = [
    os.path.join(ROOT, "ds4flash.gguf"),
    os.path.join(ROOT, "gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf"),
]
DRAFTER_CANDIDATES = [
    os.path.join(ROOT, "gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf"),
    os.path.join(ROOT, "gguf/DSpark-drafter-Q2K-Q8-0731.gguf"),
]

# Blackwell GB10 fast-path optimizations (same as the shell script).
ENV = {
    "DS4_CUDA_FP4_INDEX": "1",
    "DS4_CONT_CAPTURE": "1",
}

PERF_CORES = "5-9,15-19"  # Grace Cortex-X925 performance cores.


def first_existing(paths):
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def run_bench(model, prompt_file, extra_args):
    """Run ds4-bench and return parsed CSV rows (list of dicts)."""
    cmd = []
    if shutil_which("taskset"):
        cmd += ["taskset", "-c", PERF_CORES]
    cmd += [BENCH, "--cuda", "-m", model, "--prompt-file", prompt_file] + extra_args

    env = dict(os.environ)
    env.update(ENV)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    # Find the CSV header line and parse everything after it.
    lines = out.splitlines()
    header_idx = next((i for i, l in enumerate(lines) if l.startswith("ctx_tokens,")), None)
    if header_idx is None:
        print(out)
        return []
    reader = csv.DictReader(lines[header_idx:])
    return list(reader)


def shutil_which(name):
    try:
        import shutil
        return shutil.which(name)
    except Exception:
        return None


def main():
    prompt_file = None
    model = None
    drafter = None
    args = sys.argv[1:]
    if "-h" in args or "--help" in args:
        print(__doc__)
        return 0
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--prompt-file":
            prompt_file = args[i + 1]; i += 2
        elif a == "--model":
            model = args[i + 1]; i += 2
        elif a == "--drafter":
            drafter = args[i + 1]; i += 2
        else:
            print(f"unknown arg: {a}")
            return 1
    if prompt_file is None:
        prompt_file = os.path.join(ROOT, "speed-bench/promessi_sposi.txt")
    if model is None:
        model = first_existing(MODEL_CANDIDATES)
    if drafter is None:
        drafter = first_existing(DRAFTER_CANDIDATES)

    if not model or not os.path.isfile(model):
        print("Errore: modello DeepSeek V4 Flash non trovato.")
        print("Scaricalo con: ./download_model.sh ds4f-q2")
        return 1
    if not os.path.isfile(prompt_file):
        print(f"Errore: prompt file non trovato: {prompt_file}")
        return 1

    print("=" * 60)
    print("  NVIDIA DGX Spark (SM121) Benchmark Suite")
    print(f"  Base Model: {model}")
    if drafter:
        print(f"  Drafter:    {drafter} (DSpark attivo)")
    print("=" * 60)

    def show(title, rows):
        print("\n>>> " + title + " <<<")
        if not rows:
            print("  (nessun dato CSV)")
            return
        hdr = ["ctx_tokens", "prefill_tps", "gen_tps", "gen_first_ms", "gen_steady_tps"]
        print(f"{'ctx':>8} {'prefill_tps':>11} {'gen_tps':>9} {'ttft_ms':>9} {'steady_tps':>10}")
        for r in rows:
            ctx = int(r["ctx_tokens"])
            ptps = float(r["prefill_tps"]) if r["prefill_tps"] else 0.0
            gtps = float(r["gen_tps"]) if r["gen_tps"] else 0.0
            ttft = float(r["gen_first_ms"]) if r["gen_first_ms"] else 0.0
            stps = float(r["gen_steady_tps"]) if r["gen_steady_tps"] else 0.0
            print(f"{ctx:8d} {ptps:11.1f} {gtps:9.1f} {ttft:9.1f} {stps:10.1f}")

    # 1. Pure prefill sweep (D2R MoE tensor cores).
    show("Benchmark PREFILL (D2R MoE Tensor Cores)",
         run_bench(model, prompt_file,
                   ["--ctx-start", "2048", "--ctx-max", "16384",
                    "--step-incr", "4096", "--gen-tokens", "0"]))

    # 2. Plain single-token decode (bandwidth-bound roofline).
    show("Benchmark PLAIN DECODE (Memoria LPDDR5X)",
         run_bench(model, prompt_file,
                   ["--ctx-start", "2048", "--ctx-max", "8192",
                    "--step-incr", "2048", "--gen-tokens", "128"]))

    # 3. DSpark speculative decode (if drafter present).
    if drafter:
        show("Benchmark DSPARK SPECULATIVE DECODE",
             run_bench(model, prompt_file,
                       ["--dspark", "--mtp-model", drafter,
                        "--ctx-start", "2048", "--ctx-max", "4096",
                        "--step-incr", "2048", "--gen-tokens", "128"]))
    else:
        print("\n>>> Benchmark DSPARK SPECULATIVE DECODE <<<")
        print("  (drafter non trovato: ./download_model.sh ds4f-dspark)")

    print("\nBenchmark completato con successo!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
