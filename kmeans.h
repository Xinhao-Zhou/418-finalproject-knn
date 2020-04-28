//
// Created by 郭三山 on 4/23/20.
//

#ifndef INC_418_FINALPROJECT_KNN_KMEANS_H
#define INC_418_FINALPROJECT_KNN_KMEANS_H

#define DEFAULT_K 8
#define MAX_ITERATION 300
#define EUCLIDEAN 0
#define CONVERGE_THRESHOLD 1e-4
#include "knn_header.h"
class Cluster{
public:
    Cluster(){

    }

    DataPoint centralPoint;
    vector<DataPoint> pointList;
    int size;
private:

protected:
};

class Kmeans{
public:
    Kmeans(){
        this->k = DEFAULT_K;
    }

    Kmeans(int k){
        this->k = k;
    }
    vector<Cluster> clusters;
    int k;
private:
protected:
};

Kmeans clustersInit(vector<DataPoint> points, int k);
vector<DataPoint> getNearestClusterSet(DataPoint testInput, Kmeans clusters);

#endif //INC_418_FINALPROJECT_KNN_KMEANS_H
