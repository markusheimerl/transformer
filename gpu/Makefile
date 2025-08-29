CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda -Wno-unknown-cuda-version
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublas

train.out: transformer.o attention.o mlp.o data.o train.o
	$(CC) transformer.o attention.o mlp.o data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

transformer.o: transformer.c transformer.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c transformer.c -o $@

attention.o: ../attention/gpu/attention.c ../attention/gpu/attention.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../attention/gpu/attention.c -o $@

mlp.o: ../mlp/gpu/mlp.c ../mlp/gpu/mlp.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../mlp/gpu/mlp.c -o $@

data.o: ../attention/data.c ../attention/data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../attention/data.c -o $@

train.o: train.c transformer.h ../attention/data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin *.attn *.mlp
	$(MAKE) -C ../mlp clean
	$(MAKE) -C ../attention clean