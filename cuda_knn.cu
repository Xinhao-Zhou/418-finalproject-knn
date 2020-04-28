#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cuda_knn.h"

#define SHARED_SIZE 3060
#define TRAIN_SIZE 100
#define TEST_SIZE 10
/*
ComputeDistance: Compute the distances between test instances and training instances.
				Save the distance in device_distances

*/
__global__ void kernelComputeDistance(double *trainAttr, double *testAttr, 
	double* device_distances, int trainSize, int testSize, int attrSize){
	//48KB shared array

	//Warp size: 32. the Number of instances should fit the warp size.
	__shared__ double trainData[SHARED_SIZE];//Number of attributes X Number of Train instances in this batch
	__shared__ double testData[SHARED_SIZE];//Number of attributes X Number of Test instances in this batch

	int trainIdx = threadIdx.x;
	int testIdx = threadIdx.y;

	int trainOffset = blockDim.x * blockIdx.x;
	int testOffset = blockDim.y * blockIdx.y;

	trainIdx += trainOffset;
	testOffset += testOffset;

	//Each thread compute a distance of x to y.


    //Read train data
    //Threads that need the same train instance will read it together
	for(int i = 0;i < attrSize;i += blockDim.y){
		int tmpIdx = i + threadIdx.y;
		if(tmpIdx < attrSize){
			trainData[threadIdx.x * attrSize + threadIdx.y + i * blockDim.y] = 
			trainAttr[trainIdx * attrSize + threadIdx.y + i * blockDim.y];

			testData[threadIdx.y * attrSize + threadIdx.y + i * blockDim.y] = 
			testAttr[testIdx * attrSize + threadIdx.y + i * blockDim.y];
		}
	}


	double distance = 0.f;
	//Compute distances
	for(int i = 0;i < attrSize;i++){
		distance += trainData[threadIdx.x]
	}
}


int *cuPredict(double *trainAttr, int* trainLabels, int trainSize, int trainLength, 
	double *testAttr, int testSize, int testLength, int attrSize){

	double *device_trainAttr, device_testAttr, device_trainLabels, device_distances;
	cudaMalloc((void **)&device_trainAttr, sizeof(double) * trainSize);
	cudaMalloc((void **)&device_trainLabels, sizeof(int) * trainLength);
	cudaMalloc((void **)&device_testAttr, sizeof(double) * testSize);
	cudaMalloc((void **)&device_distances, sizeof(double) * testLength * trainLength);

	cudaMemcpy(device_trainAttr, trainAttr, sizeof(double) * trainSize, cudaMemcpyHostToDevice);
	cudaMemcpy(device_trainLabels, trainLabels, sizeof(int) * trainLength, cudaMemcpyHostToDevice);
	cudaMemcpy(device_testAttr, testAttr, sizeof(double) * testSize, cudaMemcpyHostToDevice);




	cudaFree(device_trainAttr);
	cudaFree(device_testAttr);
	cudaFree(device_trainLabels);

	//Get distance
	//Sort distance
	//find nearest neighbor
	//return labels
}