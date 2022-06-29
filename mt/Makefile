ifndef config
  config=release
endif

ifndef verbose
  SILENT = @
endif

ifndef CXX
  CXX = g++
endif

build:
	@echo "==== Building mt ===="
	mkdir -p bin/
	gcc -o bin/mt.exe mt.c mt19937ar.h mt19937ar.c
