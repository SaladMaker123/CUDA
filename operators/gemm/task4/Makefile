
NVCC := nvcc

NVCCFLAGS := -O3 -std=c++17 -arch=sm_120 -lineinfo

CONFIGS := \
	128-128-8-8-8 \
	128-128-16-8-8 \
	256-256-16-8-8

BINS := $(addprefix sgemm-,$(addsuffix .x,$(CONFIGS)))

.PHONY: all clean

all: results/out.log

define BUILD_SGEMM
sgemm-$(1).x: sgemm.cu utils.h
	$(NVCC) $(NVCCFLAGS) $$< -o $$@ \
		-DBM=$(word 1,$(subst -, ,$(1))) \
		-DBN=$(word 2,$(subst -, ,$(1))) \
		-DBK=$(word 3,$(subst -, ,$(1))) \
		-DTM=$(word 4,$(subst -, ,$(1))) \
		-DTN=$(word 5,$(subst -, ,$(1)))
endef

$(foreach cfg,$(CONFIGS),$(eval $(call BUILD_SGEMM,$(cfg))))

results/out.log: $(BINS)
	@mkdir -p results
	@: > $@
	@set -e; for bin in $(BINS); do \
		echo "Running $$bin"; \
		./$$bin >> $@; \
	done

clean:
	rm -f $(BINS)
