DEBUG=0
CC=g++ 
NVCC=nvcc
CFLAGS= -O3 -Wall -g  -std=c++11 -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC_FLAGS=-O3 -m64 --gpu-architecture compute_61 --std=c++11#-ccbin /usr/bin/$(CC) --compiler-options "-std=c++11"
OMP=-fopenmp -DOMP
LDFLAGS= -lm

CFILES = main.cpp knn_prep.cpp kmeans.cpp
HFILES = knn_header.h kmeans.h

all: knn

cuda: cudaKmeans.o predict.o

knn: knn_prep.o kmeans.o cycletimer.o cudaKmeans.o predict.o cudaKnn.o
	$(CC) $(CFLAGS) main.cpp -o knn  cudaKmeans.o cycletimer.o predict.o cudaKnn.o kmeans.o knn_prep.o

ompknn: knn_prep.o cycletimer.o
	$(CC) -O3 -Wall -g -std=c++11 $(OMP) omp_knn.cpp -o omp_knn knn_prep.cpp cycletimer.o $(LDFLAGS)

knn_prep.o:
	$(CC) $(CFLAGS) -c knn_prep.cpp -o knn_prep.o

kmeans.o: knn_prep.o
	$(CC) $(CFLAGS) -c kmeans.cpp -o kmeans.o

cycletimer.o:
	$(CC) $(CFLAGS) -c cycletimer.c -o cycletimer.o

cudaKmeans.o:
	$(NVCC) $(NVCC_FLAGS) -c cuda_kmeans.cu -o cudaKmeans.o

cudaKnn.o: cudaKmeans.o
	$(NVCC) $(NVCC_FLAGS) -c cuda_knn.cu -o cudaKnn.o -l cudaKmeans.o

predict.o:
	$(NVCC) $(NVCC_FLAGS) -c predict.cu -o predict.o

clean:
	rm -f knn knn_prep.o kmeans.o cycletimer.o cudaKmeans.o predict.o cudaKnn.o omp_knn
