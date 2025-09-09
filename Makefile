CC = clang
CFLAGS = -O3 -march=native -ffast-math -Wall -Wextra
LDFLAGS = -lopenblas -lm -flto

train.out: transformer.o attention/attention.o mlp/mlp.o data.o train.o
	$(CC) transformer.o attention/attention.o mlp/mlp.o data.o train.o $(LDFLAGS) -o $@

transformer.o: transformer.c transformer.h
	$(CC) $(CFLAGS) -c transformer.c -o $@

attention/attention.o:
	$(MAKE) -C attention attention.o

mlp/mlp.o:
	$(MAKE) -C mlp mlp.o

data.o: data.c data.h
	$(CC) $(CFLAGS) -c data.c -o $@

train.o: train.c transformer.h data.h
	$(CC) $(CFLAGS) -c train.c -o $@

run: train.out
	@time ./train.out

clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C gpu clean
	$(MAKE) -C mlp clean
	$(MAKE) -C attention clean