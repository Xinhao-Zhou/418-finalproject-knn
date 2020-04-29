//
// Created by 19612 on 2020/4/27.
//

#ifndef INC_418_FINALPROJECT_KNN_CUDA_KMEANS_H
#define INC_418_FINALPROJECT_KNN_CUDA_KMEANS_H

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

};

class cudaKmeans{
public:
	cudaKmeans(){

	}

	cudaCluster *clusters;
	int *pointClusterIdx;
protected:

private:	
};

void getClusters(double *trainSet, int trainSize, int attributesCount, int k);

#endif //INC_418_FINALPROJECT_KNN_CUDA_KMEANS_H
