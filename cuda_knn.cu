#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cuda_knn.h"

#define MAXATTRSIZE 8
#define TRAIN_SIZE 128
#define TEST_SIZE 128
/*
ComputeDistance: Compute the distances between test instances and training instances.
				Save the distance in device_distances

*/
__global__ void kernelComputeDistance(double *trainAttr, double *testAttr, 
	double* device_distances, int trainSize, int testSize, int attrSize){

	// __shared__ double trainData[MAXATTRSIZE * TRAIN_SIZE];//Number of attributes X Number of Train instances in this batch
	// __shared__ double testData[MAXATTRSIZE * TEST_SIZE];//Number of attributes X Number of Test instances in this batch

	int trainIdx = threadIdx.x;
	int testIdx = threadIdx.y;

	int trainOffset = blockDim.x * blockIdx.x;
	int testOffset = blockDim.y * blockIdx.y;

	trainOffset += trainOffset;
	testOffset += testOffset;

	//Each thread compute a distance of x to y.


    //Read train data
    //Threads that need the same train instance will read it together
    if(trainOffset < trainSize && testOffset < testSize){
    	double distance = 0.f;
    	for(int i = 0;i < attrSize;i++){
    		double trainAttribute = trainAttr[trainOffset * attrSize + i];

			double testAttribute = testAttr[testOffset * attrSize + i];
			distance += pow(trainAttribute - testAttribute, 2);
		}
		device_distances[testOffset * trainSize + trainOffset] = distance;
	}
}


__global__ initializeIndex(int *device_index, int trainSize){
	int testOffset = blockDim.y * blockIdx.y;
	int trainOffset = blockDim.x * blockIdx.x;

	testOffset += threadIdx.y;
	trainOffset += threadIdx.x;

	device_index[testOffset * trainSize + trainOffset] = trainOffset;
}
int *cuPredict(double *trainAttr, int* trainLabels, int trainSize, 
	double *testAttr, int testSize, int attrSize, int k){

	double *device_trainAttr, *device_testAttr, *device_trainLabels, *device_distances;
	int *device_index;
	cudaMalloc((void **)&device_trainAttr, sizeof(double) * trainSize * attrSize);
	cudaMalloc((void **)&device_trainLabels, sizeof(int) * trainSize);
	cudaMalloc((void **)&device_index, sizeof(int) * trainSize * testSize);
	cudaMalloc((void **)&device_testAttr, sizeof(double) * testSize * attrSize);
	cudaMalloc((void **)&device_distances, sizeof(double) * trainSize * testSize);

	cudaMemcpy(device_trainAttr, trainAttr, sizeof(double) * trainSize * attrSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_trainLabels, trainLabels, sizeof(int) * trainSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_testAttr, testAttr, sizeof(double) * testSize * attrSize, cudaMemcpyHostToDevice);

	int blockdimY = (testSize + TEST_SIZE - 1) / TEST_SIZE;
	int blockdimX = (trainSize + TRAIN_SIZE - 1) / TRAIN_SIZE;
	dim3 gridDim(blockdimX, blockdimY);
	dim3 blockDim(TRAIN_SIZE, TEST_SIZE); 

	kernelComputeDistance<<<gridDim, blockDim>>>(device_trainAttr, device_testAttr, 
		device_distances, trainSize,testSize, attrSize);

	initializeIndex<<<gridDim, blockDim>>>(device_index, trainSize);
	cudaDeviceSynchronize();	
	for(int i = 0;i < testSize;i++){
		thrust::sort_by_key(device_distance + i * trainSize, device_distance + (i + 1) * trainSize, device_index + (i * trainSize));
	}

	cudaFree(device_distances);
	cudaFree(device_index);
	cudaFree(device_trainAttr);
	cudaFree(device_testAttr);
	cudaFree(device_trainLabels);

	//Get distance
	//Sort distance
	//find nearest neighbor
	//return labels
}