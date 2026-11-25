
NVCC := nvcc

NVCCFLAGS := -O3 -std=c++17 -arch=sm_120 -lineinfo

LDLIBS := -lcublas

COMMON := ../../../common/cuda_check.h \
          ../../../common/cublas_check.h

VERSIONS := $(basename $(wildcard sgemm-v*.cu))

BINS := $(addsuffix .x,$(VERSIONS)) cublas_benchmark.x

.PHONY: all clean

all: $(BINS)

%.x: %.cu utils.h $(COMMON)
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

clean:
	rm -f $(BINS)
