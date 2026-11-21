NVCC = nvcc
LIBS = -lcufft -lcublas -lcurand -lcusparse
DEBUG = -g
# code version stamped into the binary (git commit, or VERSION file if present)
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || cat VERSION 2>/dev/null || echo unknown)
GPU_SOURCES = $(wildcard src/*.cu)
GPU_OBJECTS = $(GPU_SOURCES:.cu=.o)

all: $(GPU_OBJECTS) 
	$(NVCC) -o wnCH_NS $(GPU_OBJECTS) $(LIBS)

$(GPU_OBJECTS): src/%.o: src/%.cu 
	$(NVCC) -DGIT_COMMIT=\"$(GIT_COMMIT)\" -c   $< -o $@

clean:   
	rm src/*.o wnCH_NS *.txt *.bin *.out
