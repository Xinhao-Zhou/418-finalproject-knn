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
#define BLOCK_DIM 64
#define MAX_K 32

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

__global__ void assignLabel(double *device_distances, int *device_index, int *device_label, int *labels, int k, int testSize ,int trainSize){
	//k * testSize total threads
	__shared__ int sharedLabel[MAX_K * BLOCK_DIM];
	__shared__ int sharedSize[MAX_K * BLOCK_DIM];
	__shared__ double sharedDistance[MAX_K * BLOCK_DIM];

	int labelSize = 0;
	int testOffset = blockIdx.x * BLOCK_DIM + threadIdx.x;

	if(testOffset >= testSize)return;
	//Load data
	for(int i = 0;i < k;i++){
		int idx = device_index[testOffset * trainSize + i];
		int newLabel = device_label[idx];
		double distance = device_distances[idx];

		//check contain
		int containFlag = -1;
		for(int j = 0;j < labelSize;j++){
			if(sharedLabel[threadIdx.x * k + j] == newLabel){
				containFlag = j;
				break;
			}
		}
		if(containFlag != -1){
			sharedDistance[threadIdx.x * k + containFlag] += distance;
			sharedSize[threadIdx.x * k + containFlag] += 1;
		}else{
			sharedLabel[threadIdx.x * k + labelSize] = newLabel;
			sharedDistance[threadIdx.x * k + labelSize] = distance;
			sharedSize[threadIdx.x * k + labelSize] = 1;
			labelSize++;
		}
	}

        

	double minDistance = INFINITY;
        int maxSize = 0;
	int minlabel = -1;
	for(int i = 0;i < labelSize;i++){
		int offset = threadIdx.x * k + i;
//		double averageDistance = sharedDistance[offset] / static_cast<double>(sharedSize[offset]);
		int size = sharedSize[offset];
		if(size > maxSize){
			//minDistance = averageDistance;
			maxSize = size;
			minlabel = sharedLabel[offset];
		}
	}

	labels[testOffset] = minlabel;
}

int *cuPredict(double *trainAttr, int* trainLabels, int trainSize, 
	double *testAttr, int testSize, int attrSize, int k){
	double *device_trainAttr, *device_testAttr, *device_distances;

    int *device_trainLabels;
	int *device_testLabels;
	int *device_index;


	cudaMalloc((void **)&device_trainAttr, sizeof(double) * trainSize * attrSize);
	cudaMalloc((void **)&device_trainLabels, sizeof(int) * trainSize);
	cudaMalloc((void **)&device_index, sizeof(int) * trainSize * testSize);
	cudaMalloc((void **)&device_testAttr, sizeof(double) * testSize * attrSize);
	cudaMalloc((void **)&device_distances, sizeof(double) * trainSize * testSize);
	cudaMalloc((void **)&device_testLabels, sizeof(int) * testSize);

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


	// double *h_distances = new double[trainSize * testSize];
	// int *h_idx = new int[trainSize * testSize];

    device_distances = thrust::raw_pointer_cast(keys);
    device_index = thrust::raw_pointer_cast(vals);
	//cudaMemcpy(h_distances, device_distances, sizeof(double) * trainSize * testSize, cudaMemcpyDeviceToHost);


	int *retLabels = new int[testSize];
	assignLabel<<<(testSize + BLOCK_DIM - 1 / BLOCK_DIM), BLOCK_DIM>>>(device_distances, device_index, 
		device_trainLabels, device_testLabels, k, testSize, trainSize);

	cudaMemcpy(retLabels, device_testLabels, sizeof(int) * testSize, cudaMemcpyDeviceToHost);


//         for(int i = 0;i < attrSize;i++){
// //		printf("attr: %lf\n", testAttr[i]);

// 	}
// 	for(int i = 0;i < trainSize;i++){
// 		printf("%lf\n",h_distances[i]);
// 	}	

	cudaFree(device_distances);
	cudaFree(device_testLabels);
	cudaFree(device_index);
	cudaFree(device_trainAttr);
	cudaFree(device_testAttr);
	cudaFree(device_trainLabels);

	return retLabels;
	//Get distance
	//Sort distance
	//find nearest neighbor
	//return labels
}

/* test ver of parallel computation of different clusters */

__global__ void kernelComputeDistanceII(double *trainAttr, double *testAttr, 
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


__global__ void kernelComputeDistanceII(double *trainAttr, double *testAttr, 
	double* device_distances, int attrSize, int clusterNumber,
	int *trainClusterSize, int *testClusterSize, int maxTrainClusterSize,
	int maxTestClusterSize){

	int trainSize = 0;
	int testSize = 0;

	int maxClusterSize = maxTrainClusterSize * maxTestClusterSize;

	trainSize = trainClusterSize[blockIdx.x];
	testSize = testClusterSize[blockIdx.x];

	//Offset within a cluster, used to detect overflow
	int testOffsetInCluster = blockIdx.y * blockDim.y + threadIdx.x;
	int trainOffsetInCluster = blockIdx.z * blockDim.z + threadIdx.y;

	//Overall offset within trainAttr and testAttr
	int trainOffset = (trainOffsetInCluster + blockIdx.x * maxTrainClusterSize) * attrSize;
	int testOffset = (testOffsetInCluster + blockIdx.x * maxTestClusterSize) * attrSize;

	if(trainOffsetInCluster < trainSize && testOffsetInCluster < testSize){
		//compute distance
		double distance = 0.f;
		for(int i = 0;i < attrSize;i++){
			distance += pow(trainAttr[trainOffset + i] - testAttr[testOffset + i], 2);
		}
		distance = sqrt(distance);
		device_distances[maxClusterSize * blockIdx.x + 
		(blockIdx.y * blockDim.y + threadIdx.x) * maxTrainClusterSize + 
		blockIdx.z * blockDim.z + threadIdx.y]
			= distance;
	}
}

__global__ void initializeIndexII(int *device_index, int maxTrainClusterSize, int maxTestClusterSize,
	int *clusterTrainSize, int *clusterTestSize){
	int trainSize = 0;
	int testSize = 0;

	int maxClusterSize = maxTrainClusterSize * maxTestClusterSize;

	trainSize = clusterTrainSize[blockIdx.x];
	testSize = clusterTestSize[blockIdx.x];

	//Offset within a cluster, used to detect overflow
	int trainOffsetInCluster = blockIdx.z * blockDim.z + threadIdx.z;
	int testOffsetInCluster = blockIdx.y * blockDim.y + threadIdx.y;

	//Overall offset within trainAttr and testAttr
	int trainOffset = trainOffsetInCluster + blockIdx.x * maxTrainClusterSize;
	int testOffset = testOffsetInCluster + blockIdx.x * maxTestClusterSize;

	if(trainOffsetInCluster < trainSize && testOffsetInCluster < testSize){
		device_index[maxClusterSize * blockIdx.x + testOffsetInCluster * maxTrainClusterSize + trainOffsetInCluster] = 
		trainOffsetInCluster;
	}
}

__global__ void assignLabelII(double *device_distances, int *device_index, 
	int *device_label, int *labels, int k, int *testSize ,int *trainSize,
	int maxTrainClusterSize, int maxTestClusterSize){

	__shared__ int sharedLabel[MAX_K * BLOCK_DIM];
	__shared__ int sharedSize[MAX_K * BLOCK_DIM];

	int maxClusterSize = maxTrainClusterSize * maxTestClusterSize;

	//offset in device_distances, device_index, which is (clusterIdx, testIdx, trainIdx)
	int testOffset = maxClusterSize * blockIdx.x + (blockIdx.y * BLOCK_DIM + threadIdx.x) * maxTrainClusterSize;
	int testOffsetInCluster = blockIdx.y * BLOCK_DIM + threadIdx.x;

	//offset in shared memory
	int sharedOffset = threadIdx.x * MAX_K;

	int clusterTestSize = testSize[blockIdx.x];

//If trainSize < k, there will be a bug here. 
	int labelSize = 0;
	if(testOffsetInCluster < clusterTestSize){
		for(int i = 0;i < k;i++){
			int tmpOffset = testOffset + i;
			int idx = device_index[tmpOffset];
			int newLabel = device_label[blockIdx.x * maxTrainClusterSize + idx];

			int containFlag = -1;
			for(int j = 0;j < labelSize;j++){
				if(sharedLabel[sharedOffset + j] == newLabel){
					containFlag = j;
					break;
				}
			}

			if(containFlag != -1){
				sharedSize[sharedOffset + containFlag] += 1;
			}else{
				sharedLabel[sharedOffset + labelSize] = newLabel;
				sharedSize[sharedOffset + labelSize] = 1;
				labelSize++;
			}
		}

		int maxSize = 0;
		int maxLabel = -1;

		for(int i = 0;i < labelSize;i++){
			int tmpOffset = sharedOffset + i;
			if(sharedSize[tmpOffset] > maxSize){
				maxSize = sharedSize[tmpOffset];
				maxLabel = sharedLabel[tmpOffset];
			}
		}

		labels[blockIdx.x * maxTestClusterSize + blockIdx.y * BLOCK_DIM + threadIdx.x] = maxLabel;
	}
	

}


int *cuPredictBasedOnKmeans(cudaKmeans ckmeans, int trainSize, int testSize, int attributesCount, int k, int clusterNumber){
	//allocate memory for trainset and testSet
	//Also need offset array to record the start point and end point!
	double *device_trainAttributes, *device_testAttributes;
	double *device_distance;
	int *device_trainSize, *device_testSize;
	int *device_index;
	int *device_trainLabel;
	int *device_predictLabel;

	int maxTrainClusterSize = 0;
	int maxTestClusterSize = 0;

	for(int i = 0;i < clusterNumber;i++){
		if(ckmeans.clusters[i].size > maxTrainClusterSize){
			maxTrainClusterSize = ckmeans.clusters[i].size;
		}

		if(ckmeans.clusters[i].testSize > maxTestClusterSize){
			maxTestClusterSize = ckmeans.clusters[i].testSize;
		}
	}


	cudaMalloc((void**)&device_index, sizeof(int) * clusterNumber * maxTrainClusterSize * maxTestClusterSize);
	cudaMalloc((void**)&device_distance, sizeof(double) * clusterNumber * maxTrainClusterSize * maxTestClusterSize);
	cudaMalloc((void**)&device_trainAttributes, sizeof(double) * attributesCount * maxTrainClusterSize * clusterNumber);
	cudaMalloc((void**)&device_testAttributes, sizeof(double) * attributesCount * maxTestClusterSize * clusterNumber);
	cudaMalloc((void**)&device_trainSize, sizeof(int) * clusterNumber);
	cudaMalloc((void**)&device_testSize, sizeof(int) * clusterNumber);
	cudaMalloc((void**)&device_trainLabel, sizeof(int) * clusterNumber * maxTrainClusterSize);
	cudaMalloc((void**)&device_predictLabel, sizeof(int) * clusterNumber * maxTestClusterSize);


	//Get clusters offset
	for(int i = 0; i < clusterNumber;i++){
		cudaMemcpy(device_trainSize + i, &(ckmeans.clusters[i].size), sizeof(int),cudaMemcpyHostToDevice);
		cudaMemcpy(device_testSize + i,  &(ckmeans.clusters[i].testSize), sizeof(int), cudaMemcpyHostToDevice);
	}
	//Copy data from host to device
	for(int i = 0; i < clusterNumber;i++){
		cudaMemcpy(device_trainAttributes + i * maxTrainClusterSize * attributesCount, 
			ckmeans.clusters[i].attributes, sizeof(double) * ckmeans.clusters[i].size * attributesCount,
			cudaMemcpyHostToDevice);

		cudaMemcpy(device_testAttributes + i * maxTestClusterSize * attributesCount,
			ckmeans.clusters[i].testAttr, sizeof(double) * ckmeans.clusters[i].testSize * attributesCount,
			cudaMemcpyHostToDevice);

		cudaMemcpy(device_trainLabel + i * maxTrainClusterSize, ckmeans.clusters[i].trainLabel,
			sizeof(int) * ckmeans.clusters[i].size, cudaMemcpyHostToDevice);
	}

	//grid : (x, y, z) ==> (index of clusters, block index of test set, block index of train set)
	//block : (x, y) ==> (test offset within a block, train offset within a block)
	int blockDimX = clusterNumber;
	int blockDimY = (maxTestClusterSize + TEST_SIZE - 1) / TEST_SIZE;
	int blockDimZ = (maxTrainClusterSize + TRAIN_SIZE - 1) / TRAIN_SIZE;
	dim3 gridDim(blockDimX, blockDimY, blockDimZ);
	dim3 blockDim(TEST_SIZE, TRAIN_SIZE);

	//Compute distances
	kernelComputeDistanceII<<<gridDim, blockDim>>>(device_trainAttributes, device_testAttributes,
		device_distance, attributesCount, clusterNumber, device_trainSize, device_testSize,
		maxTrainClusterSize, maxTestClusterSize);


	initializeIndexII<<<gridDim, blockDim>>>(device_index, maxTrainClusterSize, maxTestClusterSize,
		device_trainSize, device_testSize);
	cudaDeviceSynchronize();
cudaError_t err = cudaPeekAtLastError();
printf("%s\n",cudaGetErrorName(err));


/* Sort distances and index */



    thrust::device_ptr<double> keys(device_distance);
    thrust::device_ptr<int> vals(device_index);
	
    int maxClusterSize = maxTrainClusterSize * maxTestClusterSize;


	for(int i = 0;i < clusterNumber;i++){
		int tmpTrainSize = ckmeans.clusters[i].size;
		for(int j = 0;j < ckmeans.clusters[i].testSize; j++){

			int offset = i * maxClusterSize + j * maxTrainClusterSize;//Offset of the test point

			thrust::sort_by_key(keys + offset, keys + offset + tmpTrainSize, vals + offset);
		}
				
	}

    device_distance = thrust::raw_pointer_cast(keys);
    device_index = thrust::raw_pointer_cast(vals);

/* Assign labels */
	//grid -> (x, y) -> (index of cluster, block offset)
	//thread -> (x) -> (threadOffset)
	//Every thread is responsible for a test point
	int blockY = (maxTestClusterSize + BLOCK_DIM - 1) / BLOCK_DIM; 
	dim3 assignGrid(clusterNumber, blockY);

	assignLabelII<<<assignGrid, BLOCK_DIM>>>(device_distance, device_index, 
		device_trainLabel, device_predictLabel, 
		k, device_testSize, device_trainSize, maxTrainClusterSize,
		maxTestClusterSize);

	int *retLabel = new int[testSize];
	int aggrSize = 0;
	for(int i = 0;i < clusterNumber;i++){
		int offset = i * maxTestClusterSize;
		cudaMemcpy(retLabel + aggrSize, device_predictLabel + offset, 
			sizeof(int) * ckmeans.clusters[i].testSize, cudaMemcpyDeviceToHost);
		aggrSize += ckmeans.clusters[i].testSize;
	}


	cudaFree(device_index);
	cudaFree(device_distance);
	cudaFree(device_trainAttributes);
	cudaFree(device_testAttributes);
	cudaFree(device_trainSize);
	cudaFree(device_testSize);
	cudaFree(device_trainLabel);
	cudaFree(device_predictLabel);
	
	return retLabel;
}
