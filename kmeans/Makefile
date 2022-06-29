# A simple Makefile

all: main

main: main.o randomness.o
	mkdir -p bin
	g++ -O2 -lm -o bin/kmeans.exe main.o randomness.o

main.o: main.cc master.h kmeans.h triple.h
	g++ -O2 -c -std=gnu++0x main.cc randomness.cpp

clean:
	rm -f *.o bin/kmeans.exe
