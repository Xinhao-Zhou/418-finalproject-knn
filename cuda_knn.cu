#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/host_vector.h>
#include "cuda_knn.h"

#define MAXATTRSIZE 8
#define TRAIN_SIZE 16
#define TEST_SIZE 16
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

	trainOffset += threadIdx.x;
	testOffset += threadIdx.y;

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
		device_distances[testOffset * trainSize + trainOffset] = sqrt(distance);
	}
}


__global__ void initializeIndex(int *device_index, int trainSize, int testSize){
	int testOffset = blockDim.y * blockIdx.y;
	int trainOffset = blockDim.x * blockIdx.x;

	testOffset += threadIdx.y;
	trainOffset += threadIdx.x;
	if(trainOffset < trainSize && testOffset < testSize){
		device_index[testOffset * trainSize + trainOffset] = trainOffset;
	}
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

	initializeIndex<<<gridDim, blockDim>>>(device_index, trainSize,testSize);
	cudaDeviceSynchronize();

        thrust::device_ptr<double> keys(device_distances);
        thrust::device_ptr<int> vals(device_index);
	
	for(int i = 0;i < testSize;i++){
//                 printf("here!\n");
		thrust::sort_by_key(keys + i * trainSize, keys + (i + 1) * trainSize, vals + i * trainSize);		
	}

	double *h_distances = new double[trainSize * testSize];
	int *h_idx = new int[trainSize * testSize];

        device_distances = thrust::raw_pointer_cast(keys);
	cudaMemcpy(h_distances, device_distances, sizeof(double) * trainSize * testSize, cudaMemcpyDeviceToHost);

        for(int i = 0;i < attrSize;i++){
//		printf("attr: %lf\n", testAttr[i]);

	}
	for(int i = 0;i < trainSize;i++){
		printf("%lf\n",h_distances[i]);
	}	

	cudaFree(device_distances);

	cudaFree(device_index);
	cudaFree(device_trainAttr);
	cudaFree(device_testAttr);
	cudaFree(device_trainLabels);

	int *ret = new int[10];
	return ret;
	//Get distance
	//Sort distance
	//find nearest neighbor
	//return labels
}
