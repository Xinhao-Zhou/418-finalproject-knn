#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "exclusiveScan.cu_inl"

#define BLOCK_DIM 256
#define MAXATTRSIZE 8
#define MAX_K 30
#define ITERATION_THRESHOLD 300
#define ERROR_THRESHOLD 1e-3

//Init a cuda kmeans
cudaKmeans cudaKmeansInit(int k, int attributesCount, int trainSize, double *trainSet){
	cudaKmeans ret = new cudaKmeans();

	cudaMalloc((void **)&(ret.clusters), sizeof(cudaCluster) * k);

	cudaMalloc((void) **)&(ret.pointClusterIdx, sizeof(int) * trainSize);

	for(int i = 0;i < k; i++){
		cudaMalloc((void **)&(ret.clusters[i].device_attributes), sizeof(double) * trainSize * attributesCount);
		cudaMalloc((void **)&(ret.clusters[i].centralPoint), sizeof(double) * attribtuesCount);
		cudaMalloc((void **)&(ret.clusters[i].oldCentralPoint), sizeof(double) * attribtuesCount);

		cudaMemset(&(ret.clusters[i].size), 0, sizeof(int));
		cudaMemset(&(ret.clusters[i].attributesCount), attributesCount, sizeof(int));
	}


    srand(k);
    int *randIdxList = new int[k];
    //Get initial central points
	for(int i = 0; i < k; i++){
		int r = rand() % trainSize;
		for(int j = 0;j < i;j++){
			if(randIdxList[j] == r){
				//duplicate random number
				i--;
				break;
			}

			if(j == i - 1){
				randIdxList[i] = r;
				//Copy initial central point.
				cudaMemcpy(ret.clusters[i].centralPoint, trainSet + r, sizeof(double) * attributesCount);
			}
		}
	}

	delete [] randIdxList;
	return ret;
}

double *MoveTrainSetToCuda(double *trainSet, int trainSize, int attributesCount){
	double *ret;

	cudaMalloc((void **)&(ret), sizeof(double) * trainSize * attributesCount);
	cudaMemcpy(ret, trainSet, sizeof(double) * trainSize * attributesCount, cudaMemcpyHostToDevice);

	return ret;
}

__device__ int getError(double **oldCentralPoints, double **newCentralPoints, int k, int attribtuesCount){

}

__device__ void kmeansIter(){

}

__device__ double distanceFunc(double *attr1, double *attr2, int attributesCount){
	double distance = 0;
	for(int i = 0;i < attributesCount;i++){
		distance += pow(attr1[i] - attr2[i], 2);
	}
	distance = sqrt(distance);
	return distance;
}

__global__ firstComputeDistance(cudaKmeans kmeans, double *device_trainSet, int k, int trainSize, int attributesCount){

	__shared__ double centralPoints[MAX_K * MAXATTRSIZE];
	int idx = BLOCK_DIM * blockIdx.x + threadIdx.x;
	double attr[MAXATTRSIZE];
	double minDistance = INFINITY;
	int minIdx = 0;

	if(threadIdx.x < k){
		for(int i = 0;i < attributesCount;i++){
			centralPoints[MAXATTRSIZE * threadIdx.x + i] = kmeans.clusters[threadIdx.x].centralPoints[i];
		}
	}
     
    __syncthreads();
	
	for(int i = 0;i < attributesCount;i++){
		//Read initial attributes
		attr[i] = device_trainSet[idx + i];
	}

	for(int i = 0;i < k;i++){
		double distance = distanceFunc(attr, centralPoints + i * MAXATTRSIZE, attibutesCount);
		if(distance < minDistance){
			minDistance = distance;
			minIdx = i;
		}
	}

	__syncthreads();
	//Write cluster idx to global memory.
	kmeans.pointClusterIdx[BLOCK_DIM * blockIdx.x + threadIdx.x] = minIdx;
}

//
__global__ void KmeansUpdateCentralPointsAttributes(cudaKmeans kmeans, double *device_trainSet, int k, int trainSize, int attributesCount){
	__shared__ double sumArray[BLOCK_DIM * MAXATTRSIZE];
	__shared__ double sumOutput[BLOCK_DIM];
	__shared__ double sumScratch[BLOCK_DIM * 2];

	__shared__ int inClusterFlag[BLOCK_DIM];
	__shared__ int inClusterOutput[BLOCK_DIM];
	__shared__ int inClusterScratch[BLOCK_DIM * 2];

	__shared__ double oldCentralPoint[MAXATTRSIZE * MAX_K];
	__shared__ double newCentralPoint[MAXATTRSIZE * MAX_K];

    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int clusterIdx = blockIdx.x;

    int clusterIdx = kmeans.pointClusterIdx[pointIdx];


    //Set central points' attributes to 0. Store original central points.
    if(threadIdx.x < MAXATTRSIZE * MAX_K){
    	int tmpCId = threadIdx.x / MAXATTRSIZE;
    	int attrID = threadIdx.x % MAXATTRSIZE;

    	oldCentralPoint[tmpCId * MAXATTRSIZE + attrID] = kmeans.cluster[tmpCId].centralPoint[attrID];
    	kmeans.cluster[tmpCId].centralPoint[attrID] = 0.f;//Set the original central point to 0
   	}

   	//Set cluster size to 0.
   	if(threadIdx.x < k){
   		kmeans.cluster[threadIdx.x].size = 0;
   	}

   	__syncthreads();

    for(int i = 0;i < k;i++){
    	if(i == clusterIdx){
    		inClusterFlag[threadIdx.x] = 1;
    	}else{
    		inClusterFlag[threadIdx.x] = 0;
    	}

    	__syncthreads();
    	//Do prefix sum
    	sharedMemExclusiveScanInt(threadIdx.x, inClusterFlag, inClusterOutput, inClusterScratch, BLOCK_DIM);

    	if(threadIdx.x == BLOCK_DIM - 1){
    		//Add cluster size
    		inClusterOutput[threadIdx.x] += inClusterFlag[threadIdx.x];
    		atomicAdd(&(kmeans.clusters[i].size), inClusterOutput[threadIdx.x]);//Remember to set this to 0!
    	}
    }

	for(int i = 0;i < k; i++){
		if(i == clusterIdx){
			for(int j = 0;j < attributesCount;j++){
				sumArray[BLOCK_DIM * j + threadIdx.x] = device_trainSet[(pointIdx + threadIdx.x) * attributesCount + j];		
			}
		}else{
			for(int j = 0;j < attributesCount;j++){
				sumArray[BLOCK_DIM * j + threadIdx.x] = 0.f;		
			}
		}

		__syncthreads();
		//Sum all attributes inside this block
		for(int j = 0;j < attributesCount;j++){
			//Save the last one before the prefix sum.
			if(threadIdx.x == BLOCK_DIM - 1){
				double tmp = sumArray[(j + 1) * BLOCK_DIM - 1];
			}

			sharedMemExclusiveScan(threadIdx.x, sumArray + j * BLOCK_DIM, sumOutput, sumScratch, BLOCK_DIM);

			if(threadIdx.x == BLOCK_DIM - 1){
				//Add the last element
				sumOutput[threadIdx.x] += tmp;

				//Add to global variable
				atomicAdd(&(kmeans.clusters[i].centralPoint[j]), sumOutput[threadIdx.x]);
			}
			__syncthreads();
		}
	}


	// //Get new central points
	// if(threadIdx.x < k * attributesCount){
	// 	int row = threadIdx.x / attributesCount;
	// 	int col = threadIdx.x % attributesCount;
	// 	newCentralPoint[row * attributesCount + col] = kmeans.clusters[row].centralPoint[col] / kmeans.clusters;
	// }
}

__global__ void KmeansGetNewCentralPoint(cudaKmeans kmeans, int k, int attributesCount){
	__shared__ int sizes[MAX_K];
	__shared__ double newCentralPoints[MAX_K * MAXATTRSIZE];

	if(threadIdx.x < k){
		sizes[threadIdx.x] = kmeans.clusters[threadIdx.x].size;
	}

	__syncthreads();

	if(threadIdx.x < k * attributesCount){
		int row = threadIdx.x / attributesCount;
		int col = threadIdx.x % attributesCount;

		newCentralPoints[row * attributesCount + col] = kmeans.clusters[row].centralPoint[col] / sizes[row];
		kmeans.clusters[row].centralPoint[col] = newCentralPoints[row * attributesCount + col];//write back
	}

}

__global__ void compareOldAndNewCentralPoint(cudaKmeans kmeans, int *quitFlag, int iteration, int k, int attributesCount){
	if(threadIdx.x == 0){
		*quitFlag = 0;
	}

	__syncthreads();

	if(iteration > ITERATION_THRESHOLD){
		*quitFlag = 1;
		return;
	}

	if(threadIdx.x < k * attributesCount){
		int row = threadIdx.x / attributesCount;
		int col = threadIdx.x % attributesCount;

		double diff = fabs(kmeans.clusters[row].centralPoint[col] - kmeans.clusters[row].oldCentralPoint[col]);
		if(diff > ERROR_THRESHOLD){
			atomicAdd(quitFlag, 1);
		}
		kmeans.clusters[row].oldCentralPoint[col] = kmeans.clusters[row].centralPoint[col];//Set old central point.
	}
}

cudaKmeans getClusters(double *trainSet, int trainSize, int attributesCount, int k){
	double *device_trainSet;

	cudaKmeans ret = cudaKmeansInit(k, attributesCount, trainSize);
	device_trainSet = MoveTrainSetToCuda(trainSet, trainSize, attributesCount);


}