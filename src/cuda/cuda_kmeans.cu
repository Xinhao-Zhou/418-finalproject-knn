#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>
#include "cuda_kmeans.h"
#include "exclusiveScan.cu_inl"
#include "cycletimer.h"
#include <stdio.h>

#define BLOCK_DIM 256
#define MAXATTRSIZE 8
#define MAX_K 16
#define ITERATION_THRESHOLD 300
#define ERROR_THRESHOLD 0.01f

//Init a cuda kmeans
/*
cudaKmeans *cudaKmeansInit(int k, int attributesCount, int trainSize, double *trainSet){
	cudaKmeans *ret = new cudaKmeans();

	cudaMalloc((void **)&(ret->clusters), sizeof(cudaCluster) * k);

	cudaMalloc((void **)&(ret->pointClusterIdx), sizeof(int) * trainSize);

	for(int i = 0;i < k; i++){
		cudaMalloc((void **)&(ret->clusters[i].device_attributes), sizeof(double) * trainSize * attributesCount);
		cudaMalloc((void **)&(ret->clusters[i].centralPoint), sizeof(double) * attributesCount);
		cudaMalloc((void **)&(ret->clusters[i].oldCentralPoint), sizeof(double) * attributesCount);

		cudaMemset(&(ret->clusters[i].size), 0, sizeof(int));
		cudaMemset(&(ret->clusters[i].attributesCount), attributesCount, sizeof(int));
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
				cudaMemcpy(ret->clusters[i].centralPoint, trainSet + r, sizeof(double) * attributesCount, cudaMemcpyHostToDevice);
			}
		}
	}

	delete [] randIdxList;
	return ret;
}
*/
double *MoveTrainSetToCuda(double *trainSet, int trainSize, int attributesCount){
	double *ret;

	cudaMalloc((void **)&(ret), sizeof(double) * trainSize * attributesCount);

	cudaMemcpy(ret, trainSet, sizeof(double) * trainSize * attributesCount, cudaMemcpyHostToDevice);

	return ret;
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

__global__ void firstComputeDistance(double *centralPoint, int *pointClusterIdx, double *device_trainSet, int k, int trainSize, int attributesCount){

	__shared__ double centralPoints[MAX_K * MAXATTRSIZE];
	int idx = BLOCK_DIM * blockIdx.x + threadIdx.x;
	double attr[MAXATTRSIZE];
	double minDistance = INFINITY;
	int minIdx = 0;

	if(threadIdx.x < k){
		for(int i = 0;i < attributesCount;i++){
			centralPoints[MAXATTRSIZE * threadIdx.x + i] = centralPoint[attributesCount * threadIdx.x + i];
		}
	}
     
    __syncthreads();
        if(idx < trainSize){	
	for(int i = 0;i < attributesCount;i++){
		//Read initial attributes
		attr[i] = device_trainSet[idx * attributesCount + i];
	}

	for(int i = 0;i < k;i++){
		double distance = distanceFunc(attr, centralPoints + i * MAXATTRSIZE, attributesCount);
		if(distance < minDistance){
			minDistance = distance;
			minIdx = i;
		}
	}
        }
	__syncthreads();
	//Write cluster idx to global memory.
	pointClusterIdx[BLOCK_DIM * blockIdx.x + threadIdx.x] = minIdx;

}



__global__ void KmeansUpdateCentralPointsAttributes(int iteration, double *centralPoint, int *clusterSize, int *pointClusterIdx, double *device_trainSet, int k, int trainSize, int attributesCount){
	__shared__ double sumArray[MAX_K * BLOCK_DIM];
	__shared__ double sumOutput[BLOCK_DIM];
	__shared__ double sumScratch[BLOCK_DIM * 2];

	__shared__ uint inClusterFlag[BLOCK_DIM];
	__shared__ uint inClusterOutput[BLOCK_DIM];
	__shared__ uint inClusterScratch[BLOCK_DIM * 2];

//	__shared__ double oldCentralPoint[MAXATTRSIZE * MAX_K];
//	__shared__ double newCentralPoint[MAXATTRSIZE * MAX_K];
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
 
    int clusterIdx = -1;
    if(pointIdx < trainSize)clusterIdx = pointClusterIdx[pointIdx];
 //               if(iteration == 1 && blockIdx.x == 0)
  //              printf("block :%d cluster: %d \n", blockIdx.x, clusterIdx);

    //Set central points' attributes to 0. Store original central points.

    	//oldCentralPoint[tmpCId * MAXATTRSIZE + attrID] = kmeans->clusters[tmpCId].centralPoint[attrID];
//    	centralPoint[attrID] = 0.f;//Set the original central point to 0
   	
/*
   	//Set cluster size to 0.
   	if(threadIdx.x < k){
   		clusterSize[threadIdx.x] = 0;
   	}
   	__syncthreads();
*/
    for(int i = 0;i < k;i++){

        inClusterOutput[2 * threadIdx.x] = 0;
        inClusterScratch[2 * threadIdx.x] = 0;
        inClusterScratch[2 * threadIdx.x + 1] = 0;
    	if(i == clusterIdx){
    		inClusterFlag[threadIdx.x] = 1;
      //          if(iteration == 1 && blockIdx.x == 0)
    //            printf("cluster here: %d \n", i);
    	}else{
    		inClusterFlag[threadIdx.x] = 0;
    	}

    	__syncthreads();
    	//Do prefix sum
    	sharedMemExclusiveScanInt(threadIdx.x, inClusterFlag, inClusterOutput, inClusterScratch, BLOCK_DIM);

    	if(threadIdx.x == BLOCK_DIM - 1){
    		//Add cluster size
    		inClusterOutput[threadIdx.x] += inClusterFlag[threadIdx.x];

//if(iteration == 1 && blockIdx.x == 0)                
//printf("sum:  %d\n",inClusterOutput[threadIdx.x]);
    		atomicAdd(&(clusterSize[i]), inClusterOutput[threadIdx.x]);//Remember to set this to 0!

    	}
    }
	for(int i = 0;i < k; i++){
         sumOutput[threadIdx.x] = 0;
         sumScratch[2 * threadIdx.x] = 0;    
         sumScratch[2 * threadIdx.x + 1] = 0;
		if(i == clusterIdx){
			for(int j = 0;j < attributesCount;j++){
				sumArray[j * BLOCK_DIM + threadIdx.x] = device_trainSet[(pointIdx) * attributesCount + j];		
			}
		}else{
			for(int j = 0;j < attributesCount;j++){
				sumArray[j * BLOCK_DIM + threadIdx.x] = 0.f;		
			}
		}

		__syncthreads();
		//Sum all attributes inside this block
		for(int j = 0;j < attributesCount;j++){
            double tmp;
			//Save the last one before the prefix sum.
			if(threadIdx.x == BLOCK_DIM - 1){
				tmp = sumArray[(j + 1) * BLOCK_DIM - 1];

			}

//                        if(iteration == 0 && blockIdx.x == 0 && threadIdx.x == 0){
//				for(int l = 0;l < BLOCK_DIM * attributesCount;l++){
//                                   printf("%lf\n", sumArray[l]);
//                                }   
//}


			sharedMemExclusiveScan(threadIdx.x, sumArray + j * BLOCK_DIM, sumOutput, sumScratch, BLOCK_DIM);
   
__syncthreads(); 
			if(threadIdx.x == BLOCK_DIM - 1){
				//Add the last element
				sumOutput[threadIdx.x] += tmp;
				//Add to global variabl
                //                printf("block %d cluster %d %lf\n", blockIdx.x, i, sumOutput[threadIdx.x]);

				atomicAdd(&(centralPoint[i * attributesCount + j]), sumOutput[threadIdx.x]);

			}
			__syncthreads();
		}
	}
}



//
// __global__ void KmeansUpdateCentralPointsAttributes(int iteration, double *centralPoint, int *clusterSize, int *pointClusterIdx, double *device_trainSet, int k, int trainSize, int attributesCount){


// //	__shared__ double sumArray[BLOCK_DIM * MAXATTRSIZE];
// //	__shared__ double sumOutput[BLOCK_DIM];
// //	__shared__ double sumScratch[BLOCK_DIM * 2];

// //	__shared__ uint inClusterFlag[BLOCK_DIM];
// //	__shared__ uint inClusterOutput[BLOCK_DIM];
// //	__shared__ uint inClusterScratch[BLOCK_DIM * 2];

// //	__shared__ double oldCentralPoint[MAXATTRSIZE * MAX_K];
// //	__shared__ double newCentralPoint[MAXATTRSIZE * MAX_K];
//     int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
 
//     int clusterIdx = -1;
//     if(pointIdx < trainSize)clusterIdx = pointClusterIdx[pointIdx];
//     //Set central points' attributes to 0. Store original central points.

//     	//oldCentralPoint[tmpCId * MAXATTRSIZE + attrID] = kmeans->clusters[tmpCId].centralPoint[attrID];
// //    	centralPoint[attrID] = 0.f;//Set the original central point to 0
// /*
//    	//Set cluster size to 0.
//    	if(threadIdx.x < k){
//    		clusterSize[threadIdx.x] = 0;
//    	}

//    	__syncthreads();
// */
//    	if(clusterIdx != -1){
//    		atomicAdd(&(clusterSize[clusterIdx]),1);
// 	   	for(int i = 0;i < attributesCount;i++){
// 	   		atomicAdd(&centralPoint[clusterIdx * attributesCount + i], device_trainSet[pointIdx * attributesCount + i]);
// 	   	}
// 	}
// 	// for(int i = 0;i < k;i++){
//  //        inClusterOutput[2 * threadIdx.x] = 0;
//  //        inClusterScratch[2 * threadIdx.x] = 0;
//  //        inClusterScratch[2 * threadIdx.x + 1] = 0;
//  //    	if(i == clusterIdx){
//  //    		inClusterFlag[threadIdx.x] = 1;
//  //    	}else{
//  //    		inClusterFlag[threadIdx.x] = 0;
//  //    	}

//  //    	__syncthreads();
//  //    	//Do prefix sum
//  //    	sharedMemExclusiveScanInt(threadIdx.x, inClusterFlag, inClusterOutput, inClusterScratch, BLOCK_DIM);

// 	// __syncthreads();

//  //    	if(threadIdx.x == BLOCK_DIM - 1){
//  //    		//Add cluster size
//  //    		inClusterOutput[threadIdx.x] += inClusterFlag[threadIdx.x];
//  //    		atomicAdd(&(clusterSize[i]), inClusterOutput[threadIdx.x]);//Remember to set this to 0!

//  //    	}
//  //    }
// 	// for(int i = 0;i < k; i++){
//  //         sumOutput[threadIdx.x] = 0;
//  //         sumScratch[2 * threadIdx.x] = 0;    
//  //         sumScratch[2 * threadIdx.x + 1] = 0;
// 	// 	if(i == clusterIdx){
// 	// 		for(int j = 0;j < attributesCount;j++){
// 	// 			sumArray[BLOCK_DIM * j + threadIdx.x] = device_trainSet[(pointIdx) * attributesCount + j];		
// 	// 		}
// 	// 	}else{
// 	// 		for(int j = 0;j < attributesCount;j++){
// 	// 			sumArray[BLOCK_DIM * j + threadIdx.x] = 0.f;		
// 	// 		}
// 	// 	}

// 	// 	__syncthreads();
// 	// 	//Sum all attributes inside this block
// 	// 	for(int j = 0;j < attributesCount;j++){
//  //            double tmp;
// 	// 		//Save the last one before the prefix sum.
// 	// 		if(threadIdx.x == BLOCK_DIM - 1){
// 	// 			tmp = sumArray[(j + 1) * BLOCK_DIM - 1];
// 	// 		}

// 	// 		sharedMemExclusiveScan(threadIdx.x, sumArray + j * BLOCK_DIM, sumOutput, sumScratch, BLOCK_DIM);
    
// 	// 		if(threadIdx.x == BLOCK_DIM - 1){
// 	// 			//Add the last element
// 	// 			sumOutput[threadIdx.x] += tmp;
// 	// 			//Add to global variable
//  //                                centralPoint[i * attributesCount + j] += sumOutput[threadIdx.x];
// 	// 			//atomicAdd(&(centralPoint[i * attributesCount + j]), sumOutput[threadIdx.x]);
// 	// 		}
// 	// //		__syncthreads();
// 	// 	}

// 	// }

// }

__global__ void KmeansGetNewCentralPoint(double *centralPoint, int *clusterSize, int k, int attributesCount){
	__shared__ int sizes[MAX_K];
	__shared__ double newCentralPoints[MAX_K * MAXATTRSIZE];

	if(threadIdx.x < k){
		sizes[threadIdx.x] = clusterSize[threadIdx.x];
	}

	__syncthreads();

	if(threadIdx.x < k * attributesCount){
		int row = threadIdx.x / attributesCount;
		int col = threadIdx.x % attributesCount;
		newCentralPoints[row * attributesCount + col] = centralPoint[row * attributesCount + col] / sizes[row];
		centralPoint[row * attributesCount + col] = newCentralPoints[row * attributesCount + col];//write back
	}
}

__global__ void compareOldAndNewCentralPoint(double *centralPoint, double *oldCentralPoint, int *quitFlag, int iteration, int k, int attributesCount){
	__shared__ double diffs[MAX_K * MAXATTRSIZE];
/*
	if(threadIdx.x == 0){
		*quitFlag = 0;
	}
*/
/*
	if(iteration > ITERATION_THRESHOLD){
		*quitFlag = 0;
		return;
	}
*/
	if(threadIdx.x < k * attributesCount){ 
		int row = threadIdx.x / attributesCount;
		int col = threadIdx.x % attributesCount;
		int offset = row * attributesCount + col;

		double diff = fabs(centralPoint[offset] - oldCentralPoint[offset]);
                diffs[offset] = diff;
		//if(diff > ERROR_THRESHOLD){
		//	atomicAdd(quitFlag, 1);
		//}
		oldCentralPoint[offset] = centralPoint[offset];//Set old central point.
	}

        if(threadIdx.x < k){
                double distanceTmp = 0.f;
                for(int i = 0;i < attributesCount;i++){
                    distanceTmp += pow(diffs[threadIdx.x * attributesCount + i], 2);
                }
                distanceTmp = sqrt(distanceTmp);
                if(distanceTmp > ERROR_THRESHOLD){
                    atomicAdd(quitFlag, 1);
                }
        }

}

__global__ void getNewClusterCenter(double *trainSet, int k, int attributesCount, int* pointClusterIdx, double *centralPoint, int trainSize, int *clusterSize){
    __shared__ double centralPoints[MAX_K * MAXATTRSIZE];
    __shared__ double attributes[BLOCK_DIM * MAXATTRSIZE];
    int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(threadIdx.x < k * attributesCount){
        centralPoints[threadIdx.x] = centralPoint[threadIdx.x];
    }
    __syncthreads();
    double minDistance = INFINITY;
    int minIdx = -1;
    if(pointIdx < trainSize){
    for(int i = 0;i < attributesCount;i++){
        attributes[threadIdx.x * attributesCount + i] = trainSet[pointIdx * attributesCount + i];
    }
    for(int i = 0;i < k;i++){
        double distance = 0.f;
        for(int j = 0;j < attributesCount;j++){
             double tmp = attributes[threadIdx.x * attributesCount + j] - centralPoints[i * attributesCount + j];
    	     distance += pow(tmp, 2);
        }
        distance = sqrt(distance);

        if(distance < minDistance){
            minDistance = distance;
            minIdx = i;
        }
    }

    pointClusterIdx[pointIdx] = minIdx;
    atomicAdd(&(clusterSize[minIdx]),1);
    }

}

__global__ void setQuitFlag(int *quitFlag){
	*quitFlag = 0;
}

__host__ int compareOldAndNewCentralPoint(double *centralPoint, double *oldCentralPoint, int k, int attributesCount){
	for(int i = 0;i < k;i++){
		double distance = 0.f;
		for(int j = 0;j < attributesCount;j++){
			distance += pow(centralPoint[i * attributesCount + j] - oldCentralPoint[i * attributesCount + j], 2);
		}
		distance = sqrt(distance);
		if(distance > ERROR_THRESHOLD){
			return 1;
		}
	}
	return 0;
}

cudaKmeans getClusters(double *trainSet, int trainSize, int *labelTrain, int attributesCount, int k){



	double *device_trainSet;
	int *pointClusterIdx;
	double *centralPoint;
	double *oldCentralPoint;
	int *clusterSize;


	cudaMalloc((void **)&pointClusterIdx, sizeof(int) * trainSize);
	cudaMalloc((void **)&centralPoint, sizeof(double) * attributesCount * k);
	cudaMalloc((void **)&oldCentralPoint, sizeof(double) * attributesCount * k);
	cudaMalloc((void **)&clusterSize, sizeof(int) * k);
	cudaMemset(clusterSize, 0, sizeof(int) * k);
	srand(k);

        cudaMalloc((void **)&(device_trainSet), sizeof(double) * trainSize * attributesCount);
        cudaMemcpy(device_trainSet, trainSet, sizeof(double) * trainSize * attributesCount, cudaMemcpyHostToDevice);

	double *host_centralPoint = new double[k * attributesCount];
	double *host_oldCentralPoint = new double[k * attributesCount];

    double startTime = currentSeconds();


    int *tmpRandList = new int[k];
	for(int i = 0;i < k;i++){
		int tmpRand = rand() % trainSize;
		if(i == 0){
			tmpRandList[i] = tmpRand;
		}
		for(int j = 0;j < i;j++){
			if(tmpRand == tmpRandList[j]){
				i--;
				break;
			}
			if(j == i - 1){
				tmpRandList[i] = tmpRand;
			}
		}
	}

	for(int i = 0;i < k;i++){
		int idx = tmpRandList[i];
		cudaMemcpy(centralPoint + i * attributesCount, trainSet + idx * attributesCount,sizeof(double) * attributesCount, cudaMemcpyHostToDevice);
		cudaMemcpy(oldCentralPoint + i * attributesCount, trainSet+ idx * attributesCount,sizeof(double) * attributesCount, cudaMemcpyHostToDevice);
	}
	delete [] tmpRandList;
//	device_trainSet = MoveTrainSetToCuda(trainSet, trainSize, attributesCount);

	int blockCount = (trainSize + BLOCK_DIM - 1) / BLOCK_DIM;

	int *device_quitFlag;
	int iteration = 0;
	int quitFlag;
	quitFlag = 1;
	
	cudaMalloc((void**)&device_quitFlag, sizeof(int));

	firstComputeDistance<<<blockCount, BLOCK_DIM>>>(centralPoint, pointClusterIdx, device_trainSet, k, trainSize, attributesCount);

      

	for(;quitFlag > 0 && iteration < ITERATION_THRESHOLD;iteration++){

		cudaMemset(centralPoint, 0, sizeof(double) * k * attributesCount);
                cudaMemset(clusterSize, 0, sizeof(int) * k);


		KmeansUpdateCentralPointsAttributes<<<blockCount, BLOCK_DIM>>>(iteration,centralPoint, clusterSize, pointClusterIdx, device_trainSet, k, trainSize, attributesCount);


		cudaDeviceSynchronize();

		KmeansGetNewCentralPoint<<<1, BLOCK_DIM>>>(centralPoint, clusterSize, k, attributesCount);

		cudaDeviceSynchronize();

                cudaMemset(device_quitFlag, 0, sizeof(int));

		compareOldAndNewCentralPoint<<<1, BLOCK_DIM>>>(centralPoint, oldCentralPoint, device_quitFlag, iteration, k, attributesCount);
		cudaMemcpy(&quitFlag, device_quitFlag, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemset(clusterSize, 0, sizeof(int) * k);
 		getNewClusterCenter<<<blockCount, BLOCK_DIM>>>(device_trainSet, k, attributesCount, pointClusterIdx, centralPoint, trainSize, clusterSize);
		cudaDeviceSynchronize();

	}


	//Copy from device to host..


	int *host_clusterSize = new int[k];

	int *host_pointClusterIdx = new int[trainSize];



	cudaMemcpy(host_clusterSize, clusterSize, sizeof(int) * k, cudaMemcpyDeviceToHost);
	cudaMemcpy(host_pointClusterIdx, pointClusterIdx, sizeof(int) * trainSize,  cudaMemcpyDeviceToHost);

	cudaKmeans cuRet;// = new cudaKmeans();
	cuRet.clusters = new cudaCluster[k];

	for(int i = 0;i < k;i++){
		cuRet.clusters[i].attributesCount = attributesCount;
		cuRet.clusters[i].size = host_clusterSize[i];
                cuRet.clusters[i].centralPoint = new double[attributesCount];
                cuRet.clusters[i].attributes = new double[attributesCount * host_clusterSize[i]];
		cuRet.clusters[i].trainLabel = new int[host_clusterSize[i]];
		cudaMemcpy(cuRet.clusters[i].centralPoint, centralPoint + i * attributesCount, sizeof(double) * attributesCount, cudaMemcpyDeviceToHost);
	}

	int *clusterIdx = new int[k]();
	for(int i = 0;i < trainSize;i++){
		int idx = host_pointClusterIdx[i];
		for(int j = 0;j < attributesCount;j++){
			//Assign instances to clusters
			cuRet.clusters[idx].attributes[clusterIdx[idx] * attributesCount + j] = trainSet[i * attributesCount + j];			
		}
		cuRet.clusters[idx].trainLabel[clusterIdx[idx]] = labelTrain[i];
		clusterIdx[idx]++;
	}


	delete [] host_clusterSize;
	delete [] clusterIdx;
        delete [] host_pointClusterIdx;


	cudaFree(pointClusterIdx);
	cudaFree(centralPoint);
	cudaFree(oldCentralPoint);
	cudaFree(clusterSize);
	cudaFree(device_trainSet);
	cudaFree(device_quitFlag);

	double endTime = currentSeconds();

	printf("parallel kmeans time: %lf\n", endTime - startTime);

	return cuRet;
}

__global__ void assignCluster(double *centralPoint, double *testAttr, int *assignedCluster, int testSize, int k, int attributesCount){
	__shared__ double sharedCentralPoint[MAX_K * MAXATTRSIZE];
	__shared__ double sharedTestAttr[MAXATTRSIZE * BLOCK_DIM];

	int testOffset = BLOCK_DIM * blockIdx.x + threadIdx.x;

	if(threadIdx.x < k * attributesCount){
		sharedCentralPoint[threadIdx.x] = centralPoint[threadIdx.x];
	}

	__syncthreads();

	if(testOffset < testSize){
		for(int i = 0;i < attributesCount;i++){
			sharedTestAttr[threadIdx.x * attributesCount + i] = testAttr[testOffset * attributesCount + i];
		}
	}else{
		return;
	}

	double minDistance = INFINITY;
	int minIdx = 0;
	for(int i = 0;i < k;i++){
		double distance = 0.f;
		for(int j = 0;j < attributesCount;j++){
			distance += pow(sharedTestAttr[threadIdx.x * attributesCount + j] - sharedCentralPoint[i * attributesCount + j],2);
		}
		distance = sqrt(distance);
                //printf("%d clus: %d distance: %lf\n",testOffset, i, distance);
		if(distance < minDistance){
			minIdx = i;
			minDistance = distance;
		}
	}
	assignedCluster[testOffset] = minIdx;
}

cudaKmeans getClusterId(cudaKmeans ckmeans, double *testAttr, int testSize,int k, int *testLabel){

	double *device_centralPoints, *device_testAttr;
	int *device_assignedCluster;

	int *assignedCluster = new int[testSize];
	int attributesCount = ckmeans.clusters[0].attributesCount;
	int *testAssignedSize = new int[k]();

	cudaMalloc((void**)&device_centralPoints, sizeof(double) * attributesCount * k);
	cudaMalloc((void**)&device_testAttr, sizeof(double) * attributesCount * testSize);
	cudaMalloc((void**)&device_assignedCluster, sizeof(int) * testSize);

	cudaMemcpy(device_testAttr, testAttr, sizeof(double) * attributesCount * testSize, cudaMemcpyHostToDevice);

	for(int i = 0;i < k;i++){
		cudaMemcpy(device_centralPoints + i * attributesCount, ckmeans.clusters[i].centralPoint, 
			sizeof(double) * attributesCount, cudaMemcpyHostToDevice);
	}

	assignCluster<<<(testSize + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(device_centralPoints, device_testAttr, 
		device_assignedCluster, testSize, k, attributesCount);

	cudaMemcpy(assignedCluster, device_assignedCluster, sizeof(int) * testSize, cudaMemcpyDeviceToHost);


	for(int i = 0;i < testSize;i++){
		int clusterIdx = assignedCluster[i];
		testAssignedSize[clusterIdx]++;
	}
	for(int i = 0;i < k;i++){
		ckmeans.clusters[i].testAttr = new double[testAssignedSize[i] * attributesCount];
		ckmeans.clusters[i].testLabel = new int[testAssignedSize[i]];
		ckmeans.clusters[i].testSize = testAssignedSize[i];

		testAssignedSize[i] = 0;
	}


	for(int i = 0;i < testSize;i++){
		int clusterIdx = assignedCluster[i];
		int size = testAssignedSize[clusterIdx];
		for(int j = 0;j < attributesCount;j++){
			ckmeans.clusters[clusterIdx].testAttr[size * attributesCount + j] = testAttr[i * attributesCount + j];
		}
		ckmeans.clusters[clusterIdx].testLabel[testAssignedSize[clusterIdx]] = testLabel[i];
		testAssignedSize[clusterIdx]++;
	}


	cudaFree(device_centralPoints);
	cudaFree(device_testAttr);
	cudaFree(device_assignedCluster);


	delete [] assignedCluster;
	delete [] testAssignedSize;

	return ckmeans;
}


