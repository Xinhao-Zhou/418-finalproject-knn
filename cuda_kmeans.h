//
// Created by 19612 on 2020/4/27.
//

#ifndef INC_418_FINALPROJECT_KNN_CUDA_KMEANS_H
#define INC_418_FINALPROJECT_KNN_CUDA_KMEANS_H

#include "knn_header.h"
#define DEFAULT_K 8

class cudaCluster{
public:
	cudaCluster(){

	}

	double *device_attributes;
	int size;
	int attributesCount;
	double *centralPoint;//central points' attributes.
	double *oldCentralPoint;
protected:

private:	
}

class cudaKmeans{
public:
	cudaKmeans(){

	}

	cudaCluster *clusters;
	int *pointClusterIdx;
protected:

private:	
}

cudaKmeans getClusters();

#endif //INC_418_FINALPROJECT_KNN_CUDA_KMEANS_H
