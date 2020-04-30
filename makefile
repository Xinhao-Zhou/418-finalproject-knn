DEBUG=0
CC=g++ 
NVCC=nvcc
CFLAGS= -O3 -Wall -g  -std=c++11 -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC_FLAGS=-O3 -m64 --gpu-architecture compute_61 --std=c++11#-ccbin /usr/bin/$(CC) --compiler-options "-std=c++11"

CFILES = main.cpp knn_prep.cpp kmeans.cpp
HFILES = knn_header.h kmeans.h

all: knn

cuda: cudaKmeans.o

knn: knn_prep.o kmeans.o cycletimer.o cudaKmeans.o
	$(CC) $(CFLAGS) main.cpp -o knn  cudaKmeans.o cycletimer.o

knn_prep.o:
	$(CC) $(CFLAGS) -c knn_prep.cpp -o knn_prep.o

kmeans.o: knn_prep.o
	$(CC) $(CFLAGS) -c kmeans.cpp -o kmeans.o

cycletimer.o:
	$(CC) $(CFLAGS) -c cycletimer.c -o cycletimer.o

cudaKmeans.o:
	$(NVCC) $(NVCC_FLAGS) -c cuda_kmeans.cu -o cudaKmeans.o

clean:
	rm -f knn knn_prep.o kmeans.o cycletimer.o cudaKmeans.o
