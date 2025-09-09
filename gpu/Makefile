CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C ../attention clean
	$(MAKE) -C ../mlp clean