CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

ARCH ?= sm_86
CUDAFLAGS = --cuda-gpu-arch=$(ARCH) -x cuda
CUDALIBS = -L/usr/local/cuda/lib64 -lcudart -lcublasLt

train.out: transformer.o ../attention/gpu/attention.o ../mlp/gpu/mlp.o ../data.o train.o
	$(CC) transformer.o ../attention/gpu/attention.o ../mlp/gpu/mlp.o ../data.o train.o $(CUDALIBS) $(LDFLAGS) -o $@

transformer.o: transformer.c transformer.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c transformer.c -o $@

../attention/gpu/attention.o:
	$(MAKE) -C ../attention/gpu attention.o

../mlp/gpu/mlp.o:
	$(MAKE) -C ../mlp/gpu mlp.o

../data.o: ../data.c ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c ../data.c -o $@

train.o: train.c transformer.h ../data.h
	$(CC) $(CFLAGS) $(CUDAFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C ../attention/gpu clean
	$(MAKE) -C ../mlp/gpu clean