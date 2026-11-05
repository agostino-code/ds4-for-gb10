CC ?= cc
UNAME_S := $(shell uname -s)
NATIVE_CPU_FLAG ?= -march=native

SAMPLING_TEST := tests/test_sampling
GLM53_KDA_TEST := tests/test_glm53_kda
QWEN4_KERNEL_TEST := tests/test_qwen4_kernels

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99 -D_GNU_SOURCE -fno-finite-math-only -fstack-protector-strong -D_FORTIFY_SOURCE=2 -DDS4_TARGET_SM121=1 -DDS4_UNIFIED_MEMORY=1
QUALITY_CFLAGS ?= -O3 $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c11 -fstack-protector-strong -D_FORTIFY_SOURCE=2

LDLIBS ?= -lm -pthread
DS4_TEST_MODEL ?= ds4flash.gguf
DS4_TEST_MTP ?= gguf/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf
DS4_DSPARK_MODEL ?= $(DS4_TEST_MODEL)
DS4_DSPARK_SUPPORT ?= gguf/DeepSeek-V4-Flash-DSpark-support-0731.gguf

CUDA_HOME ?= $(shell if [ -x /usr/local/cuda/bin/nvcc ]; then \
	printf '%s' /usr/local/cuda; \
	elif command -v nvcc >/dev/null 2>&1; then \
	dirname "$$(dirname "$$(command -v nvcc)")"; \
	else \
	printf '%s' /usr/local/cuda; \
	fi)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_121
ifneq ($(strip $(CUDA_ARCH)),)
ifneq ($(filter sm_120 sm_120a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_120a,code=sm_120a -DDS4_CUDA_HAVE_MXF4=1
else ifneq ($(filter sm_121 sm_121a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -gencode arch=compute_121a,code=compute_121a -DDS4_CUDA_HAVE_MXF4=1 -DDS4_CUDA_FASTPATH_SPARK=1 -DDS4_TARGET_SM121=1 -DDS4_USE_D2R=1
CFLAGS += -DDS4_CUDA_FASTPATH_SPARK=1
NVCC_PTXAS_FLAGS ?= -Xptxas -v,-dlcm=ca --extra-device-vectorization
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) $(NVCC_PTXAS_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
# Vendored llama.cpp mmq prefill tier (cuda/mmq/, see cuda/mmq/VENDOR.md).
MMQ_INCLUDES := -Icuda/mmq
MMQ_OBJS := cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
CORE_OBJS = ds4.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda.o ds4_cuda_routing.o ds4_dspark_quench.o ds4_prefix_cache.o ds4_agent_hooks.o ds4_layer_pack.o ds4_engram.o $(MMQ_OBJS)
CPU_CORE_OBJS = ds4_cpu.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_cuda_routing.o ds4_dspark_quench.o ds4_prefix_cache.o ds4_agent_hooks.o ds4_layer_pack.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)

.PHONY: all help clean test dspark-acceptance dspark-verify-depth mtp-verify-depth cpu cuda cuda-spark cuda-generic cuda-regression

all: cuda-spark

help:
	@echo "ds4-for-gb10 (NVIDIA Grace Blackwell GB10 / SM121) build targets:"
	@echo "  make                     Build CUDA binaries for DGX Spark / GB10 (default)"
	@echo "  make cuda-spark          Build CUDA binaries for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA binaries for generic CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with explicit nvcc -arch value"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make test                Build and run regression test suite"
	@echo "  make dspark-verify-depth Run DSpark speculative verification smoke if support GGUF is present"
	@echo "  make mtp-verify-depth    Run legacy MTP speculative verification smoke if MTP GGUF is present"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=sm_121

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_121"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_eval_cases.o ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_eval_cases.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args_cpu.o $(CPU_CORE_OBJS) $(LDLIBS)

ds4: ds4_cli.o ds4_help.o ds4_prompt_prefix.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_eval_cases.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o ds4_gpu_args.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

gguf-tools/quality-testing/score_official.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(QUALITY_CFLAGS)) -I. -c -o $@ $<

gguf-tools/quality-testing/score_official: gguf-tools/quality-testing/score_official.o $(CORE_OBJS) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_q8_scratch.o: tests/test_cuda_q8_scratch.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_ssd_cache.o: tests/test_cuda_ssd_cache.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_embed.o: tests/test_cuda_embed.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_embed: tests/test_cuda_embed.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_ssd_batch.o: tests/test_cuda_ssd_batch.c ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_ssd_batch: tests/test_cuda_ssd_batch.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_tp.o: tests/test_cuda_tp.c ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_tp: tests/test_cuda_tp.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_tp_repack.o: tests/test_cuda_tp_repack.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_tp_repack: tests/test_cuda_tp_repack.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_pool.o: tests/test_cuda_pool.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_pool: tests/test_cuda_pool.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_dspark.o: tests/test_cuda_dspark.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_dspark: tests/test_cuda_dspark.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_verify_head.o: tests/test_cuda_verify_head.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_verify_head: tests/test_cuda_verify_head.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_shared_side.o: tests/test_cuda_shared_side.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_shared_side: tests/test_cuda_shared_side.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_grouped_q8.o: tests/test_cuda_grouped_q8.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_grouped_q8: tests/test_cuda_grouped_q8.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_v41_fusion.o: tests/test_cuda_v41_fusion.cu ds4_gpu.h ds4_deepseek41_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_v41_fusion: tests/test_cuda_v41_fusion.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_ssd_repack.o: tests/test_cuda_ssd_repack.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_ssd_repack: tests/test_cuda_ssd_repack.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_ssd_cache: tests/test_cuda_ssd_cache.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-ssd-cache
test-cuda-ssd-cache: tests/test_cuda_ssd_cache
	./tests/test_cuda_ssd_cache

tests/test_cuda_q8_rows.o: tests/test_cuda_q8_rows.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -I. -c -o $@ $<

tests/test_cuda_q8_rows: tests/test_cuda_q8_rows.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-q8-rows
test-cuda-q8-rows: tests/test_cuda_q8_rows
	./tests/test_cuda_q8_rows

tests/test_cuda_q8_pair.o: tests/test_cuda_q8_pair.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_q8_pair: tests/test_cuda_q8_pair.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-q8-pair
test-cuda-q8-pair: tests/test_cuda_q8_pair
	./tests/test_cuda_q8_pair

tests/test_cuda_reductions.o: tests/test_cuda_reductions.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_reductions: tests/test_cuda_reductions.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-reductions
test-cuda-reductions: tests/test_cuda_reductions
	./tests/test_cuda_reductions

tests/test_cuda_indexer.o: tests/test_cuda_indexer.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_indexer: tests/test_cuda_indexer.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-indexer
test-cuda-indexer: tests/test_cuda_indexer
	./tests/test_cuda_indexer

tests/test_cuda_f16_chunks.o: tests/test_cuda_f16_chunks.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_f16_chunks: tests/test_cuda_f16_chunks.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-f16-chunks
test-cuda-f16-chunks: tests/test_cuda_f16_chunks
	./tests/test_cuda_f16_chunks

tests/test_cuda_indexed_one.o: tests/test_cuda_indexed_one.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_indexed_one: tests/test_cuda_indexed_one.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-indexed-one
test-cuda-indexed-one: tests/test_cuda_indexed_one
	./tests/test_cuda_indexed_one

tests/test_cuda_hc_norm.o: tests/test_cuda_hc_norm.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_hc_norm: tests/test_cuda_hc_norm.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-hc-norm
test-cuda-hc-norm: tests/test_cuda_hc_norm
	./tests/test_cuda_hc_norm

tests/test_cuda_hc_pre.o: tests/test_cuda_hc_pre.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_hc_pre: tests/test_cuda_hc_pre.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-hc-pre
test-cuda-hc-pre: tests/test_cuda_hc_pre
	./tests/test_cuda_hc_pre

tests/test_cuda_copy_spans.o: tests/test_cuda_copy_spans.cu ds4_gpu.h ds4_gpu_copy.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_copy_spans: tests/test_cuda_copy_spans.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-copy-spans
test-cuda-copy-spans: tests/test_cuda_copy_spans
	./tests/test_cuda_copy_spans

tests/test_cuda_router_stream.o: tests/test_cuda_router_stream.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_router_stream: tests/test_cuda_router_stream.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-router-stream
test-cuda-router-stream: tests/test_cuda_router_stream
	./tests/test_cuda_router_stream

tests/test_cuda_q8_0_gemv.o: tests/test_cuda_q8_0_gemv.cu ds4_gpu.h cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_q8_0_gemv: tests/test_cuda_q8_0_gemv.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-q8-0-gemv
test-cuda-q8-0-gemv: tests/test_cuda_q8_0_gemv
	./tests/test_cuda_q8_0_gemv

tests/test_cuda_shared.o: tests/test_cuda_shared.cu ds4_gpu.h ds4_deepseek41_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_shared: tests/test_cuda_shared.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-shared
test-cuda-shared: tests/test_cuda_shared
	./tests/test_cuda_shared

tests/test_cuda_compressor.o: tests/test_cuda_compressor.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_compressor: tests/test_cuda_compressor.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-compressor
test-cuda-compressor: tests/test_cuda_compressor
	./tests/test_cuda_compressor

tests/test_cuda_attn_side.o: tests/test_cuda_attn_side.cu ds4_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_attn_side: tests/test_cuda_attn_side.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-attn-side
test-cuda-attn-side: tests/test_cuda_attn_side
	./tests/test_cuda_attn_side

tests/test_cuda_mxfp4_split.o: tests/test_cuda_mxfp4_split.c ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_mxfp4_split: tests/test_cuda_mxfp4_split.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-mxfp4-split
test-cuda-mxfp4-split: tests/test_cuda_mxfp4_split
	./tests/test_cuda_mxfp4_split

tests/test_cuda_dspark_tp_split.o: tests/test_cuda_dspark_tp_split.c ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_dspark_tp_split: tests/test_cuda_dspark_tp_split.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_dspark_tp_markov.o: tests/test_cuda_dspark_tp_markov.c ds4_gpu.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_cuda_dspark_tp_markov: tests/test_cuda_dspark_tp_markov.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-dspark-tp
test-cuda-dspark-tp: tests/test_cuda_dspark_tp_markov tests/test_cuda_dspark_tp_split \
    tests/test_cuda_shared_side tests/test_cuda_f16_chunks tests/test_cuda_topk_stream \
    tests/test_cuda_indexed_one tests/test_cuda_embed tests/test_cuda_hc_norm \
    tests/test_cuda_hc_pre tests/test_cuda_indexer tests/test_cuda_q8_pair
	./tests/test_cuda_dspark_tp_markov
	./tests/test_cuda_dspark_tp_split
	./tests/test_cuda_dspark_tp_split --mxfp4
	./tests/test_cuda_shared_side
	./tests/test_cuda_f16_chunks
	./tests/test_cuda_topk_stream
	./tests/test_cuda_indexed_one
	./tests/test_cuda_embed
	./tests/test_cuda_hc_norm
	./tests/test_cuda_hc_pre
	./tests/test_cuda_indexer
	./tests/test_cuda_q8_pair

.PHONY: test-dspark-history
test-dspark-history:
	python3 tests/test_dspark_window.py
	python3 tests/test_dspark_confidence_input.py
	python3 tests/test_dspark_eos_contract.py
	python3 tests/test_dspark_rewind.py

tests/test_cuda_topk_stream.o: tests/test_cuda_topk_stream.cu ds4_gpu.h ds4_deepseek41_gpu.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -I. -c -o $@ $<

tests/test_cuda_topk_stream: tests/test_cuda_topk_stream.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-topk-stream
test-cuda-topk-stream: tests/test_cuda_topk_stream
	./tests/test_cuda_topk_stream

tests/test_deepseek41_prefill.o: tests/test_deepseek41_prefill.c ds4.c ds4_gpu.h ds4_engram.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_deepseek41_prefill: tests/test_deepseek41_prefill.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_deepseek41_dspark.o: tests/test_deepseek41_dspark.c ds4.c ds4_deepseek41_dspark.h ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_deepseek41_dspark: tests/test_deepseek41_dspark.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_cuda_q8_scratch: tests/test_cuda_q8_scratch.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-q8-scratch
test-cuda-q8-scratch: tests/test_cuda_q8_scratch
	./tests/test_cuda_q8_scratch

tests/test_cuda_dspark_moe.o: cuda/mmq/test/test_iq2_aligned_entry.cu cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -Icuda/mmq -c -o $@ $<

tests/test_cuda_dspark_moe: tests/test_cuda_dspark_moe.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-cuda-dspark-moe
test-cuda-dspark-moe: tests/test_cuda_dspark_moe
	./tests/test_cuda_dspark_moe

cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke

tests/test_mxfp4_cuda: tests/test_mxfp4_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

test-mxfp4-cuda: tests/test_mxfp4_cuda
	./tests/test_mxfp4_cuda

tests/test_mxfp4_staged_cuda: tests/test_mxfp4_staged_cuda.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-mxfp4-staged-cuda
test-mxfp4-staged-cuda: tests/test_mxfp4_staged_cuda
	./tests/test_mxfp4_staged_cuda

tests/test_mmvq_cuda: tests/test_mmvq.cu $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-mmvq-cuda
test-mmvq-cuda: tests/test_mmvq_cuda
	./tests/test_mmvq_cuda

tests/test_qwen4_cuda.o: tests/test_qwen4_kernels.c ds4_gpu.h ds4.h
	$(CC) $(QUALITY_CFLAGS) -D_GNU_SOURCE -I. -c -o $@ $<

tests/test_qwen4_cuda: tests/test_qwen4_cuda.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-qwen4-cuda
test-qwen4-cuda: tests/test_qwen4_cuda
	./tests/test_qwen4_cuda

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_gpu_copy.h ds4_gpu_tp.h ds4_deepseek41_gpu.h ds4_deepseek41_dspark.h ds4_linux_memory.h ds4_engram.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_image.o: ds4_image.c ds4_image.h third_party/iris/jpeg.h third_party/iris/png.h
	$(CC) $(CFLAGS) -c -o $@ ds4_image.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_engram.o: ds4_engram.c ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -c -o $@ ds4_engram.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_tp.o: ds4_tp.c ds4_tp.h ds4.h ds4_ssd.h ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(CFLAGS) -c -o $@ ds4_tp.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_prompt_prefix.o: ds4_prompt_prefix.c ds4_prompt_prefix.h ds4.h
	$(CC) $(CFLAGS) -c -o $@ ds4_prompt_prefix.c

ds4_gpu_args.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4_gpu_args.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_eval_cases.o: ds4_eval_cases.c ds4_eval_cases.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval_cases.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_image.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h ds4_gpu_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_gpu_args_cpu.o: ds4_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_gpu_args.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4_eval_cases.h ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

tests/test_glm53_kda.o: tests/test_glm53_kda.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_glm53_kda.c

tests/test_glm53_vision_engine.o: tests/test_glm53_vision_engine.c ds4.h ds4_image.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ tests/test_glm53_vision_engine.c

tests/test_glm53_vision_engine: tests/test_glm53_vision_engine.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_glm53_vision_prompt.o: tests/test_glm53_vision_prompt.c ds4.h
	$(CC) $(CFLAGS) -fno-fast-math -I. -c -o $@ tests/test_glm53_vision_prompt.c

tests/test_glm53_vision_prompt: tests/test_glm53_vision_prompt.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_deepseek4_vision_image.o: tests/test_deepseek4_vision_image.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_deepseek4_vision_image: tests/test_deepseek4_vision_image.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_image_decode.o: tests/test_image_decode.c ds4_image.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_image_decode: tests/test_image_decode.o ds4_image.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(GLM53_KDA_TEST): tests/test_glm53_kda.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-glm53-kda
test-glm53-kda: $(GLM53_KDA_TEST)
	./$(GLM53_KDA_TEST)


tests/test_qwen4_vision.o: tests/test_qwen4_vision.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/test_qwen4_vision.c

tests/test_qwen4_vision: tests/test_qwen4_vision.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

.PHONY: test-qwen4-kernels test-qwen4-q2 test-qwen4-vision
test-qwen4-kernels: $(QWEN4_KERNEL_TEST)
	./$(QWEN4_KERNEL_TEST)

.PHONY: test-qwen4-q2
test-qwen4-q2: $(QWEN4_KERNEL_TEST) tests/test_qwen4_moe_mm_specialize
	DS4_TEST_QWEN4_MV_EXACT=1 ./$(QWEN4_KERNEL_TEST)
	./tests/test_qwen4_moe_mm_specialize

# DS4_QWEN4_SNAPSHOT=<HF checkpoint dir> DS4_QWEN4_MMPROJ=<mmproj.gguf> DS4_QWEN4_IMAGE=<image>
test-qwen4-vision: tests/test_qwen4_vision
	@test -n "$(DS4_QWEN4_SNAPSHOT)" -a -n "$(DS4_QWEN4_MMPROJ)" -a -n "$(DS4_QWEN4_IMAGE)" || \
	  { echo "set DS4_QWEN4_SNAPSHOT, DS4_QWEN4_MMPROJ and DS4_QWEN4_IMAGE"; exit 1; }
	python3 tests/qwen4_vision_ref.py --snapshot "$(DS4_QWEN4_SNAPSHOT)" --mmproj "$(DS4_QWEN4_MMPROJ)" --image "$(DS4_QWEN4_IMAGE)"

tests/test_linux_memory: tests/test_linux_memory.c ds4_linux_memory.h
	$(CC) $(CFLAGS) -I. -o $@ $<

.PHONY: test-linux-memory
test-linux-memory: tests/test_linux_memory
	./tests/test_linux_memory

tests/test_glm_attention.o: tests/test_glm_attention.c ds4.h ds4_gpu.h ds4_linux_memory.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

tests/test_glm_attention: tests/test_glm_attention.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-glm-attention
test-glm-attention: tests/test_glm_attention
	./tests/test_glm_attention

tests/test_cuda_tokentile.o: tests/test_cuda_tokentile.c ds4.h ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -c -o $@ $<

tests/test_cuda_tokentile: tests/test_cuda_tokentile.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

.PHONY: test-cuda-tokentile
test-cuda-tokentile: tests/test_cuda_tokentile
	./tests/test_cuda_tokentile

tests/test_cuda_tp_link.o: tests/test_cuda_tp_link.c ds4_gpu.h ds4_gpu_tp.h ds4_tp.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_cuda_tp_link: tests/test_cuda_tp_link.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_ssd_cache: tests/test_ssd_cache.c ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -I. -o $@ tests/test_ssd_cache.c ds4_ssd.c

.PHONY: test-ssd-cache
test-ssd-cache: tests/test_ssd_cache
	./tests/test_ssd_cache

tests/test_engram: tests/test_engram.c ds4_engram.c ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -o $@ tests/test_engram.c ds4_engram.c $(LDLIBS)

.PHONY: test-engram
test-engram: tests/test_engram
	./tests/test_engram

tests/test_deepseek41_gguf.o: tests/test_deepseek41_gguf.c ds4.c ds4.h ds4_engram.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_deepseek41_gguf: tests/test_deepseek41_gguf.o ds4_engram.c $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -I. -o $@ $^ $(LDLIBS)

.PHONY: test-deepseek41-gguf
test-deepseek41-gguf: tests/test_deepseek41_gguf
	./tests/test_deepseek41_gguf

tests/test_qwen4_ngrams.o: tests/test_qwen4_ngrams.c ds4.c ds4.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_ngrams: tests/test_qwen4_ngrams.o $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(LDLIBS)

.PHONY: test-qwen4-ngrams
test-qwen4-ngrams: tests/test_qwen4_ngrams
	./tests/test_qwen4_ngrams

tests/test_qwen4_memory.o: tests/test_qwen4_memory.c ds4.c ds4.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_memory: tests/test_qwen4_memory.o $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(LDLIBS)

.PHONY: test-qwen4-memory
test-qwen4-memory: tests/test_qwen4_memory
	./tests/test_qwen4_memory

tests/test_qwen4_mtp_policy.o: tests/test_qwen4_mtp_policy.c ds4.c ds4.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_mtp_policy: tests/test_qwen4_mtp_policy.o $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(LDLIBS)

.PHONY: test-qwen4-mtp-policy
test-qwen4-mtp-policy: tests/test_qwen4_mtp_policy
	./tests/test_qwen4_mtp_policy

tests/test_qwen4_ngram_state.o: tests/test_qwen4_ngram_state.c ds4.c ds4.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_ngram_state: tests/test_qwen4_ngram_state.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_qwen4_mtp_prefill.o: tests/test_qwen4_mtp_prefill.c ds4.c ds4.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_mtp_prefill: tests/test_qwen4_mtp_prefill.o $(filter-out ds4.o,$(CORE_OBJS))
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -o $@ $^ $(LDLIBS)

tests/test_qwen4_generation.o: tests/test_qwen4_generation.c ds4.c ds4.h ds4_qwen4_vision.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_generation: tests/test_qwen4_generation.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_qwen4_ple_prefetch.o: tests/test_qwen4_ple_prefetch.c ds4.c ds4.h ds4_qwen4_vision.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_qwen4_ple_prefetch: tests/test_qwen4_ple_prefetch.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)


tests/test_qwen4_prefill.o: tests/test_qwen4_prefill.c ds4.h
	$(CC) $(QUALITY_CFLAGS) -I. -c -o $@ $<

tests/test_qwen4_prefill: tests/test_qwen4_prefill.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4.o ds4_cpu.o ds4_cpu_test_hooks.o ds4_cuda_test_hooks.o ds4_cuda.o tests/test_qwen4_cuda.o tests/test_qwen4_kernels.o tests/test_qwen4_ngram_state.o tests/test_qwen4_mtp_prefill.o tests/test_qwen4_mtp_policy.o tests/test_qwen4_memory.o: ds4_qwen4_vision.h

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_gpu_copy.h ds4_gpu_tp.h ds4_gpu_mgpu.h ds4_linux_memory.h ds4_deepseek41_gpu.h ds4_deepseek41_cuda.cuh ds4_glm53_vision_gpu.cuh ds4_deepseek4_vision_gpu.cuh ds4_qwen4_cuda.cuh ds4_image.h ds4_iq2_tables_cuda.inc cuda/mmq/ds4_mmq.h
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

# Vendored mmq pieces (see cuda/mmq/VENDOR.md).  ds4_mmq.cu transitively
# pulls in mmq.cuh which has heavy template instantiation -- each piece
# compiles in its own TU and links in.
cuda/mmq/ds4_ggml_stubs.o: cuda/mmq/ds4_ggml_stubs.cu cuda/mmq/ds4_ggml_stubs.h cuda/mmq/common.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq.o: cuda/mmq/ds4_mmq.cu cuda/mmq/ds4_mmq.h cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/mmid.cuh cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_mmq_d2r.o: cuda/mmq/ds4_mmq_d2r.cu cuda/mmq/ds4_mmq_d2r.cuh cuda/mmq/mmq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/vecdotq.cuh cuda/mmq/mma.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/quantize.o: cuda/mmq/quantize.cu cuda/mmq/quantize.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/mmq.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmid.o: cuda/mmq/mmid.cu cuda/mmq/mmid.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/mmvq.o: cuda/mmq/mmvq.cu cuda/mmq/mmvq.cuh cuda/mmq/common.cuh cuda/mmq/ds4_ggml_stubs.h cuda/mmq/quantize.cuh cuda/mmq/vecdotq.cuh cuda/mmq/unary.cuh
	$(NVCC) $(NVCCFLAGS) -std=c++17 $(MMQ_INCLUDES) -c -o $@ $<

cuda/mmq/ds4_repack.o: cuda/mmq/ds4_repack.cu cuda/mmq/ds4_repack.h
	$(NVCC) $(NVCCFLAGS) -std=c++17 -c -o $@ $<

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_layer_pack.o: tests/test_layer_pack.c ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_layer_pack: tests/test_layer_pack.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_gpu_args.o: tests/test_gpu_args.c ds4_gpu_args.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -DDS4_NO_GPU -c -o $@ $<

tests/test_gpu_args: tests/test_gpu_args.o ds4_gpu_args_cpu.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ds4_cpu_test_hooks.o: ds4.c ds4.h ds4_image.h ds4_gpu.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -DDS4_TEST_HOOKS -c -o $@ ds4.c

tests/test_engine_mgpu_placement.o: tests/test_engine_mgpu_placement.c ds4.h ds4_gpu_mgpu.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_engine_mgpu_placement: tests/test_engine_mgpu_placement.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_sampling.o: tests/test_sampling.c ds4.h
	$(CC) $(CFLAGS) -fno-finite-math-only -DDS4_TEST_HOOKS -I. -c -o $@ $<

tests/test_sampling: tests/test_sampling.o ds4_cpu_test_hooks.o ds4_image.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_dspark_quench.o: tests/test_dspark_quench.c ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_dspark_quench: tests/test_dspark_quench.o ds4_dspark_quench.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_NO_GPU -I. -c -o $@ $<

tests/test_session_state: tests/test_session_state.o $(filter-out ds4_cpu.o,$(CPU_CORE_OBJS))
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

tests/test_session_state_gpu.o: tests/test_session_state.c ds4.c ds4.h ds4_gpu.h ds4_image.h ds4_tp.h
	$(CC) $(CFLAGS) -Wno-unused-function -I. -c -o $@ $<

tests/test_session_state_gpu: tests/test_session_state_gpu.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_deepseek_indexer.o: tests/test_deepseek_indexer.c ds4.c ds4.h ds4_gpu.h ds4_tp.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -Wno-unused-function -I. -c -o $@ $<

tests/test_deepseek_indexer: tests/test_deepseek_indexer.o $(filter-out ds4.o,$(CORE_OBJS))
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

tests/test_tp_link.o: tests/test_tp_link.c ds4_tp.h ds4.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_tp_link: tests/test_tp_link.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test-session-state
test-session-state: tests/test_session_state
	./tests/test_session_state

ifneq ($(UNAME_S),Darwin)
tests/test_gpu_xdev.o: tests/test_gpu_xdev.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_xdev: tests/test_gpu_xdev.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_model_cache.o: tests/test_gpu_model_cache.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_model_cache: tests/test_gpu_model_cache.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_gpu_lookup_cache_strict.o: tests/test_gpu_lookup_cache_strict.c ds4_gpu.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_gpu_lookup_cache_strict: tests/test_gpu_lookup_cache_strict.o ds4_cuda.o ds4_image.o $(MMQ_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_cuda_test_hooks.o: ds4.c ds4.h ds4_gpu.h ds4_gpu_mgpu.h ds4_gpu_tp.h ds4_layer_pack.h
	$(CC) $(CFLAGS) -Wno-unused-function -DDS4_TEST_HOOKS -I$(CUDA_HOME)/include -c -o $@ ds4.c

tests/test_engine_mgpu_refusal.o: tests/test_engine_mgpu_refusal.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_refusal: tests/test_engine_mgpu_refusal.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_mgpu_runtime.o: tests/test_engine_mgpu_runtime.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_mgpu_runtime: tests/test_engine_mgpu_runtime.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o $(filter-out ds4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_engine_correctness.o: tests/test_engine_correctness.c ds4.h ds4_gpu_mgpu.h
	$(CC) $(CFLAGS) -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_engine_correctness: tests/test_engine_correctness.o ds4_gpu_args.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

tests/test_cuda_session_batch.o: tests/test_cuda_session_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h ds4_tp.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_session_batch: tests/test_cuda_session_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o $(filter-out ds4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-session-batch: tests/test_cuda_session_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_session_batch

tests/test_cuda_mixed_batch.o: tests/test_cuda_mixed_batch.c ds4.h ds4_gpu_args.h ds4_gpu_mgpu.h ds4_tp.h
	$(CC) $(CFLAGS) -DDS4_TEST_HOOKS -I. -I$(CUDA_HOME)/include -c -o $@ $<

tests/test_cuda_mixed_batch: tests/test_cuda_mixed_batch.o ds4_cuda_test_hooks.o ds4_gpu_args.o ds4_kvstore.o rax.o $(filter-out ds4.o,$(CORE_OBJS))
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

test-cuda-mixed-batch: tests/test_cuda_mixed_batch
	DS4_TEST_MODEL="$(DS4_TEST_MODEL)" ./tests/test_cuda_mixed_batch
endif

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(DS4_LINK_LIBS)

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ ds4_agent_test.o ds4_help.o ds4_prompt_prefix.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(DS4_LINK_LIBS)

tests/test_prompt_prefix.o: tests/test_prompt_prefix.c ds4_prompt_prefix.h
	$(CC) $(CFLAGS) -I. -c -o $@ $<

tests/test_prompt_prefix: tests/test_prompt_prefix.o ds4_prompt_prefix.o
	$(CC) $(CFLAGS) -o $@ $^

.PHONY: test-frontends
test-frontends: ds4_test ds4_agent_test test-web-recovery
	./ds4_test --server
	./ds4_agent_test

.PHONY: test-web-recovery
test-web-recovery: tests/test_web_recovery
	python3 tests/test_web_recovery.py ./tests/test_web_recovery

tests/test_web_recovery: tests/test_web_recovery.c ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -Wno-unused-function -o $@ tests/test_web_recovery.c

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test mxfp4-dot-test test-session-state test-linux-memory test-engram test-web-recovery \
	tests/test_layer_pack tests/test_engine_mgpu_placement tests/test_gpu_args \
	tests/test_deepseek4_vision_image tests/test_image_decode tests/test_prompt_prefix $(SAMPLING_TEST) tests/test_dspark_quench ds4 ds4-server ds4-bench ds4-agent
	./ds4-eval --validate-cases
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test
	./tests/test_layer_pack
	./tests/test_engine_mgpu_placement
	./tests/test_gpu_args
	./tests/test_gpu_args_cli.sh
	./tests/test_prompt_prefix
	./tests/test_sampling
	./tests/test_dspark_quench
	./tests/test_deepseek4_vision_image
	./tests/test_image_decode

dspark-acceptance: ds4
	DS4_DSPARK_MODEL="$(DS4_DSPARK_MODEL)" \
	DS4_DSPARK_SUPPORT="$(DS4_DSPARK_SUPPORT)" \
	sh tests/dspark_acceptance_fixture.sh

dspark-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "dspark-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_DSPARK_SUPPORT)" ]; then \
		echo "dspark-verify-depth: skipped, missing DSpark support $(DS4_DSPARK_SUPPORT)"; \
		echo "dspark-verify-depth: run ./download_model.sh ds4f-dspark or set DS4_DSPARK_SUPPORT=FILE"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_DSPARK="$(DS4_DSPARK_SUPPORT)" ./ds4_test --dspark-verify-depth; \
	fi

mtp-verify-depth: ds4_test
	@if [ ! -f "$(DS4_TEST_MODEL)" ]; then \
		echo "mtp-verify-depth: skipped, missing model $(DS4_TEST_MODEL)"; \
	elif [ ! -f "$(DS4_TEST_MTP)" ]; then \
		echo "mtp-verify-depth: skipped, missing MTP support $(DS4_TEST_MTP)"; \
		echo "mtp-verify-depth: set DS4_TEST_MTP=FILE to a legacy support GGUF"; \
	else \
		DS4_TEST_MODEL="$(DS4_TEST_MODEL)" DS4_TEST_MTP="$(DS4_TEST_MTP)" ./ds4_test --mtp-verify-depth; \
	fi

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

mxfp4-dot-test: tests/test_mxfp4_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_mxfp4_dot tests/test_mxfp4_dot.c -lm
	./tests/test_mxfp4_dot

.PHONY: test-download-model
test-download-model:
	python3 tests/test_model_download.py

.PHONY: test-quality-api
# Only the scorer's JSON parser is needed; discard the unused engine entry point.
test-quality-api: tests/test_quality_api.c gguf-tools/quality-testing/score_official.c
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
	./tests/test_quality_api
	python3 tests/test_collect_official.py

ds4.o ds4_cpu.o ds4_agent.o ds4_agent_cpu.o ds4_server.o ds4_server_cpu.o \
ds4_test.o ds4_agent_test.o \
ds4_cpu_test_hooks.o ds4_cuda_test_hooks.o tests/test_session_state.o \
tests/test_session_state_gpu.o: ds4_tool_text.h

clean:
	rm -f tests/test_mmvq_cuda
	rm -f tests/test_qwen4_ngrams
	rm -f tests/test_qwen4_memory tests/test_qwen4_mtp_policy tests/test_qwen4_ssd_experts
	rm -f tests/test_qwen4_ngram_state
	rm -f tests/test_qwen4_mtp_prefill
	rm -f tests/test_qwen4_generation
	rm -f tests/test_qwen4_resident_arithmetic
	rm -f tests/test_qwen4_ple_prefetch
	rm -f tests/test_web_recovery
	rm -f tests/test_cuda_q8_rows
	rm -f tests/test_cuda_q8_pair
	rm -f tests/test_cuda_reductions
	rm -f tests/test_cuda_indexer
	rm -f tests/test_cuda_f16_chunks
	rm -f tests/test_cuda_indexed_one
	rm -f tests/test_cuda_hc_norm
	rm -f tests/test_cuda_hc_pre
	rm -f tests/test_cuda_copy_spans
	rm -f tests/test_cuda_router_stream
	rm -f tests/test_cuda_q8_0_gemv
	rm -f tests/test_cuda_shared
	rm -f tests/test_cuda_compressor
	rm -f tests/test_cuda_mxfp4_split
	rm -f tests/test_cuda_dspark_tp_split
	rm -f tests/test_cuda_dspark_tp_markov
	rm -f tests/test_cuda_attn_side
	rm -f tests/test_mxfp4_staged_cuda
	rm -f tests/test_cuda_topk_stream
	rm -f tests/test_cuda_ssd_cache
	rm -f tests/test_cuda_ssd_batch
	rm -f tests/test_cuda_embed
	rm -f tests/test_cuda_tp
	rm -f tests/test_cuda_tp_repack
	rm -f tests/test_cuda_pool
	rm -f tests/test_cuda_dspark
	rm -f tests/test_cuda_verify_head
	rm -f tests/test_cuda_shared_side
	rm -f tests/test_cuda_grouped_q8
	rm -f tests/test_cuda_v41_fusion
	rm -f tests/test_cuda_ssd_repack
	rm -f tests/test_deepseek41_gguf
	rm -f tests/test_deepseek41_graph tests/test_deepseek41_cli
	rm -f tests/test_deepseek41_dspark
	rm -f tests/test_deepseek41_prefill
	rm -f tests/test_cuda_q8_scratch
	rm -f tests/test_cuda_dspark_moe
	rm -f tests/test_quality_api
	rm -f tests/test_linux_memory
	rm -f tests/test_glm_attention tests/test_cuda_tokentile
	rm -f tests/test_ssd_cache tests/test_engram
	rm -f tests/test_session_state tests/test_session_state_gpu
	rm -f tests/test_deepseek_indexer
	rm -f tests/test_cuda_tp_link
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test gguf-tools/quality-testing/score_official gguf-tools/quality-testing/score_official.o tests/test_q4k_dot tests/test_mxfp4_dot tests/test_mxfp4_cuda tests/test_qwen4_moe_mm_specialize tests/test_qwen4_conv_parallel tests/test_q8_prefill_variants tests/test_glm53_kda tests/test_glm53_vision_engine tests/test_glm53_vision_prompt tests/test_deepseek4_vision_image tests/test_prompt_prefix tests/test_gpu_xdev tests/test_gpu_model_cache tests/test_gpu_lookup_cache_strict tests/test_engine_mgpu_refusal tests/test_engine_mgpu_runtime tests/test_engine_correctness tests/test_sampling tests/test_cuda_session_batch tests/test_cuda_mixed_batch tests/*.o *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
	rm -f tests/test_image_decode
	rm -f tests/test_qwen4_kernels tests/test_qwen4_cuda tests/test_qwen4_vision tests/test_qwen4_prefill

# The active tokenizer includes generated Unicode classes.
ds4.o ds4_cpu.o ds4_cpu_test_hooks.o: ds4_qwen4_unicode.inc

ds4_verify_test_hooks.o: ds4.c ds4.h ds4_deepseek41_dspark.h ds4_gpu.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -DDS4_TEST_HOOKS -Wno-unused-function -c -o $@ $<

gguf-tools/quality-testing/score_official_verify.o: gguf-tools/quality-testing/score_official.c ds4.h
	$(CC) $(filter-out -ffast-math,$(CFLAGS)) -DDS4_TEST_HOOKS -I. -c -o $@ $<

gguf-tools/quality-testing/score_official_verify: gguf-tools/quality-testing/score_official_verify.o ds4_verify_test_hooks.o $(filter-out ds4.o,$(CORE_OBJS)) rax.o ds4_gpu_args.o
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)
