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

	double *attributes;
	int *trainLabel;
	int size;
	int attributesCount;
	double *centralPoint;//central points' attributes.

	double *testAttr;
	int *testLabel;
	int testSize;
};

class cudaKmeans{
public:
	cudaKmeans(){

	}

	cudaCluster *clusters;
protected:

private:	
};

cudaKmeans getClusters(double *trainSet, int trainSize, int* trainLabel, int attributesCount, int k);
cudaKmeans getClusterId(cudaKmeans ckmeans, double *testAttr, int testSize, int k, int *labelTest);

#endif //INC_418_FINALPROJECT_KNN_CUDA_KMEANS_H
