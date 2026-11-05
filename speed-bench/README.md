## Benchmarking

Here we collect prefill and generation speed obtained with different hardware.

Run `ds4-bench` as:

```
./ds4-bench \
  -m ds4flash.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

Provide PR including your numbers if your hardware was not already tested.
Call the benchmark csv file something like `gb10.csv` or alike, so that
it is clear what hardware was used for the benchmark.

To generate an SVG graph from a CSV file:

```
python3 speed-bench/plot_speed.py speed-bench/gb10.csv --title "DGX Spark t/s"
```

The script uses only the Python standard library. By default it writes a file
next to the CSV using the `_ts.svg` suffix, such as `speed-bench/gb10_ts.svg`.

### Concurrency: many requests at once

`session_concurrency_bench` measures the engine without HTTP overhead.

```sh
make session-concurrency-bench
./speed-bench/session_concurrency_bench -m ds4flash.gguf \
  --concurrency 1,2,4,8 --ctx 4096 --gen 128 --budget-gib 28 \
  --csv /tmp/concurrency.csv
```

Streams use different corpus offsets to avoid identical expert routing.
The report includes aggregate and per-stream throughput, step latency and
prefill time. Set `--budget-gib` to bound session allocations; monitor actual
memory and swap too. `--mixed` adds a concurrent continued prefill, `--spec`
enables Qwen MTP, and `--verify` fails on a greedy-token mismatch against
serial decoding. Use the official-continuation scorer to assess quality when
different reduction orders change a close token choice.

For an interleaved comparison against a diagnostic rollback:

```sh
./speed-bench/session_concurrency_bench --ctx 4096 --concurrency 8 \
  --candidate-env DS4_QWEN4_SESSION_BATCH --candidate-value 0 --repeat 3
```

The control unsets the variable; the candidate sets it in alternating ABBA
blocks. Use only controls read at dispatch time. Keep prompts, context, memory
pressure and thermal conditions comparable, and retain generated text when
measuring speculation. `concurrency_sweep.sh` runs each cell in a fresh process.

`serve_concurrency_bench.py` measures the HTTP server, including queueing,
prompt rendering, prefix reuse and streaming:

```sh
python3 speed-bench/serve_concurrency_bench.py \
  --concurrency 8 --prompt-tokens 4096 --max-tokens 128 --requests 32 \
  --ignore-eos --json /tmp/serve-8.json
```

Start the server separately with an appropriate model, context and
`--batched-session 8`. The report includes first-token and inter-token
latency, request latency and throughput. Prompts use fresh nonces;
`--shared-prefix` tests cache reuse instead.
