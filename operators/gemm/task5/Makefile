
# =====================================
# Compiler Configuration
# =====================================
NVCC := nvcc
ARCH := sm_120

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDLIBS := -lcublas

# =====================================
# Target
# =====================================
TARGET := hgemm.x

SHELL := /bin/bash

.PHONY: all run clean

# =====================================
# Build
# =====================================
all: $(TARGET)

$(TARGET): hgemm.cu utils.h
	$(NVCC) $(NVCCFLAGS) hgemm.cu -o $(TARGET) $(LDLIBS)

# =====================================
# Run Benchmark
# =====================================
run: $(TARGET)
	mkdir -p results
	set -o pipefail; ./$(TARGET) | tee results/out.log

# =====================================
# Clean
# =====================================
clean:
	rm -f $(TARGET)
