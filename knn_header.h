#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <limits>

using namespace std;


#ifndef KNN_HEADER_H_INCLUDED
#define KNN_HEADER_H_INCLUDED

struct DataPoint{
    int id;
    char label;
};

struct Distance{
    DataPoint src_datapoint;
    DataPoint dest_datapoint;
    int distance;
};


int distanceFunc(DataPoint datapoint1, DataPoint datapoint2);
vector<DataPoint> parseFile();
vector<DataPoint> parseFile_test();
bool operator <(Distance distance_a, Distance distance_b);
priority_queue<Distance> getPriorityQueue(DataPoint target_point, vector<DataPoint> datapoints);
vector<Distance> findNeighbors(DataPoint datapoint, priority_queue<Distance> train_datapoints, int k);
DataPoint assignLabel(DataPoint target_datapoint, vector<Distance> distances);
vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k);


#endif // KNN_HEADER_H_INCLUDED
