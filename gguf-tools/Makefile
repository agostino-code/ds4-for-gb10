CC ?= cc
CXX ?= c++
UNAME_S := $(shell uname -s)
NATIVE_CPU_FLAG ?= -march=native
QUANTS_SHARED := libds4quants.so
SHARED_FLAGS := -shared
CUDA_NVCC := $(shell command -v nvcc 2>/dev/null)
CUDA_HOME ?= $(if $(wildcard /usr/local/cuda/bin/nvcc),/usr/local/cuda,$(if $(CUDA_NVCC),$(patsubst %/bin/nvcc,%,$(CUDA_NVCC)),/usr/local/cuda))
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_121

ifneq ($(filter sm_121 sm_121a,$(strip $(CUDA_ARCH))),)
NVCC_ARCH_FLAGS := -gencode arch=compute_121a,code=sm_121a -DDS4_TARGET_SM121=1
else
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif

NVCCFLAGS ?= -O3 --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
QUALITY_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
QUALITY_TARGETS := ds4.o ds4_cuda.o ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o ds4_engram.o ds4_image.o rax.o ds4_gpu_args.o \
	cuda/mmq/ds4_ggml_stubs.o cuda/mmq/ds4_mmq.o cuda/mmq/ds4_mmq_d2r.o \
	cuda/mmq/quantize.o cuda/mmq/mmid.o cuda/mmq/mmvq.o cuda/mmq/ds4_repack.o
QUALITY_OBJS := ../ds4.o ../ds4_cuda.o ../ds4_distributed.o ../ds4_tp.o ../ds4_ssd.o ../ds4_layer_pack.o ../ds4_engram.o ../ds4_image.o ../rax.o ../ds4_gpu_args.o \
	../cuda/mmq/ds4_ggml_stubs.o ../cuda/mmq/ds4_mmq.o ../cuda/mmq/ds4_mmq_d2r.o \
	../cuda/mmq/quantize.o ../cuda/mmq/mmid.o ../cuda/mmq/mmvq.o ../cuda/mmq/ds4_repack.o
QUALITY_LINK := $(NVCC) $(NVCCFLAGS) -I.. -o

CFLAGS ?= -O3 -Wall -Wextra -std=c11 $(NATIVE_CPU_FLAG) -fstack-protector-strong -D_FORTIFY_SOURCE=2
CPPFLAGS ?= -D_GNU_SOURCE

.PHONY: all clean quants-shared quality-score quality-llama-score

all: deepseek4-quantize $(QUANTS_SHARED)

quants-shared: $(QUANTS_SHARED)

deepseek4-quantize: deepseek4-quantize.c quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ deepseek4-quantize.c quants.c -lm -pthread

$(QUANTS_SHARED): quants.c quants.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SHARED_FLAGS) -fPIC -o $@ quants.c -lm -pthread

quality-score:
	$(MAKE) -C .. $(QUALITY_TARGETS)
	$(QUALITY_LINK) quality-testing/score_official quality-testing/score_official.c $(QUALITY_OBJS) $(QUALITY_LDLIBS)

quality-llama-score:
	$(CXX) $(LLAMA_CPP_CXXFLAGS) -o quality-testing/score_llama quality-testing/score_llama.cpp $(LLAMA_CPP_LDLIBS)

clean:
	rm -f deepseek4-quantize libds4quants.dylib libds4quants.so \
		quality-testing/score_official quality-testing/score_llama
