SEQ_DIR=src/seq
CUDA_DIR=src/cuda
OMP_DIR=src/omp
LIB_DIR=src/lib
OBJ_DIR=src/obj

DEBUG=0
CC=g++ 
NVCC=nvcc
<<<<<<< HEAD
CFLAGS= -O3 -Wall -g  -std=c++11 -I $(LIB_DIR) -I $(SEQ_DIR) -I $(CUDA_DIR) -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC_FLAGS=-O3 -m64 --gpu-architecture compute_61 --std=c++11 -I $(LIB_DIR)
OMP= -O3 -Wall -g -std=c++11 -fopenmp -DOMP -I $(LIB_DIR)
LDFLAGS= -lm

LIB_OBJS= $(LIB_DIR)/obj/cycletimer.o $(LIB_DIR)/obj/knn_prep.o
CUDA_OBJS= $(CUDA_DIR)/obj/cuda_kmeans.o $(CUDA_DIR)/obj/cuda_knn.o $(CUDA_DIR)/obj/predict.o
SEQ_OBJS= $(SEQ_DIR)/obj/kmeans.o 
EXEC= knn
OMP_EXEC= ompknn

all: $(EXEC)

$(EXEC): $(LIB_OBJS) $(CUDA_OBJS) $(SEQ_OBJS)
	$(CC) $(CFLAGS) $(SEQ_DIR)/main.cpp -o $@ $^

$(OMP_EXEC): $(LIB_OBJS)
	$(CC) $(OMP) $(OMP_DIR)/omp_knn.cpp -o $@ $^ $(LDFLAGS)

$(LIB_DIR)/obj/%.o: $(LIB_DIR)/%.cpp 
	$(CC) $(CFLAGS) -c $^ -o $@

$(CUDA_DIR)/obj/%.o: $(CUDA_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) -c $^ -o $@
	
$(SEQ_DIR)/obj/%.o: $(SEQ_DIR)/%.cpp
	$(CC) $(CFLAGS) -c $^ -o $@
#knn_prep.o:
#	$(CC) $(CFLAGS) -c knn_prep.cpp -o knn_prep.o
=======
CFLAGS= -O3 -Wall -g  -std=c++11 -L/usr/local/depot/cuda-10.2/lib64/ -lcudart
NVCC_FLAGS=-O3 -m64 --gpu-architecture compute_61 --std=c++11#-ccbin /usr/bin/$(CC) --compiler-options "-std=c++11"
>>>>>>> master

#kmeans.o: knn_prep.o
#	$(CC) $(CFLAGS) -c kmeans.cpp -o kmeans.o

#cycletimer.o:
#	$(CC) $(CFLAGS) -c cycletimer.c -o cycletimer.o

<<<<<<< HEAD
#cudaKmeans.o:
#	$(NVCC) $(NVCC_FLAGS) -c cuda_kmeans.cu -o cudaKmeans.o
=======
cuda: cudaKmeans.o

knn: knn_prep.o kmeans.o cycletimer.o cudaKmeans.o
	$(CC) $(CFLAGS) main.cpp -o knn  cudaKmeans.o cycletimer.o
>>>>>>> master

#cudaKnn.o: cudaKmeans.o
#	$(NVCC) $(NVCC_FLAGS) -c cuda_knn.cu -o cudaKnn.o -l cudaKmeans.o

#predict.o:
#	$(NVCC) $(NVCC_FLAGS) -c predict.cu -o predict.o

cycletimer.o:
	$(CC) $(CFLAGS) -c cycletimer.c -o cycletimer.o

cudaKmeans.o:
	$(NVCC) $(NVCC_FLAGS) -c cuda_kmeans.cu -o cudaKmeans.o

clean:
<<<<<<< HEAD
	rm -f $(EXEC) $(OMP_EXEC) $(LIB_OBJS) $(CUDA_OBJS)
=======
	rm -f knn knn_prep.o kmeans.o cycletimer.o cudaKmeans.o
>>>>>>> master
