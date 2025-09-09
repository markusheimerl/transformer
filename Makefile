clean:
	rm -f *.out *.o *.csv *.bin
	$(MAKE) -C gpu clean
	$(MAKE) -C mlp clean
	$(MAKE) -C attention clean