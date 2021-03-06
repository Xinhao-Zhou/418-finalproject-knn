#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <limits>
#include <fstream>
#include <string.h>
#include <cmath>
#include <stdlib.h>
#include "cycletimer.h"

using namespace std;

#ifndef KNN_HEADER_H_INCLUDED
#define KNN_HEADER_H_INCLUDED

class DataPoint
{
public:
    DataPoint(){};
    //virtual ~DataPoint();
//    DataPoint(DataPoint const & dp){
//        this->id = dp.id;
//        for(int i = 0;i < dp.attributes.size();i++){
//            double tmp = dp.attributes[i];
//            this->attributes.push_back(tmp);
//        }
//        this->label = dp.label;
//    }
    int id;
    vector<double> attributes;
    int label;
protected:

private:
};

struct Distance{
    DataPoint src_datapoint;
    DataPoint dest_datapoint;
    double distance;
};

int *getLabelArray(vector<DataPoint> vec);
double *getAttributesArray(vector<DataPoint> vec);

double distanceFunc(DataPoint datapoint1, DataPoint datapoint2, int func);
vector<DataPoint> parseFile(int argc, char *argv[]);
vector<DataPoint> parseFile_test(int argc, char *argv[]);
bool operator <(Distance distance_a, Distance distance_b);
priority_queue<Distance> getPriorityQueue(DataPoint target_point, vector<DataPoint> datapoints, int func);
vector<Distance> findNeighbors(DataPoint datapoint, priority_queue<Distance> train_datapoints, int k);
Distance *getSmallestDistances(DataPoint datapoint, DataPoint *data_train, int k, int func);
void assignLabel_seq(DataPoint *target_datapoint, Distance *distances, int k);
int assignLabel(double *distances, int *ids, int *labels, int k);
vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func);
DataPoint *sort_datapoint(DataPoint target_datapoint, vector<DataPoint> data_train, int k, int func);
double *getDistances(DataPoint dp, DataPoint *datapoints, int train_len, int func);
int *changeVectorToArray_ID(DataPoint *datapoints, int size);
double *changeVectorToArray_Attributes(DataPoint *datapoints, int size, int attribute_len);
int *changeVectorToArray_Labels(DataPoint *datapoints, int size);
DataPoint *getKthSmallestDatapoint(DataPoint *datapoints, int k);
Distance *getKthSmallestDistance(DataPoint *k_smallest, DataPoint target_datapoint, int k, int func);
double *getKthSmallestDatapoint_Attributes(double *attributes, int attribute_len, int k);
vector<DataPoint> predictLables_seq(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func);

#endif // KNN_HEADER_H_INCLUDED
