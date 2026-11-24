# =====================================
# Compiler Configuration
# =====================================
NVCC := nvcc
ARCH := sm_120

NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
LDLIBS := -lcublas -lcuda

# =====================================
# Version Configuration
# =====================================
VERSION ?= 1

TARGET := hgemm-v$(VERSION).x

SHELL := /bin/bash

.PHONY: all run clean

# =====================================
# Build
# =====================================
all: $(TARGET)

hgemm-v%.x: hgemm-v%.cu utils.h Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

# =====================================
# Run Benchmark
# =====================================
run: $(TARGET)
	mkdir -p results
	set -o pipefail; ./$(TARGET) | tee results/out-v$(VERSION).log

# =====================================
# Clean
# =====================================
clean:
	rm -f hgemm-v*.x