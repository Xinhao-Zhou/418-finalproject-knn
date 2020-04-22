#include "..\knn_header.h"

int distanceFunc(DataPoint datapoint1, DataPoint datapoint2){
    int N = rand() % 10+1;
    return N;
}

vector<DataPoint> parseFile(){
    vector<DataPoint> pq;
    for(int i=0;i<5;i++){
        DataPoint dp;
        dp.id = i;
        dp.label = 'A' + rand()%24;
        pq.push_back(dp);
    }
    return pq;
}

vector<DataPoint> parseFile_test(){
    vector<DataPoint> pq;
    for(int i=0;i<5;i++){
        DataPoint dp;
        dp.id = i+10;
        dp.label = 'A' + rand()%24;
        pq.push_back(dp);
    }
    return pq;
}



bool operator <(Distance distance_a, Distance distance_b)
{
    if(distance_a.distance>distance_b.distance){
        return true;
    }else if(distance_a.distance==distance_b.distance){
        if(distance_a.dest_datapoint.id>distance_b.dest_datapoint.id){
            return true;
        }else{
            return false;
        }
    }else{
        return false;
    }
}

priority_queue<Distance> getPriorityQueue(DataPoint target_point, vector<DataPoint> datapoints){
    priority_queue<Distance> pq;
    for(DataPoint dp:datapoints){
        Distance ds;
        ds.src_datapoint = target_point;
        ds.dest_datapoint = dp;
        ds.distance = distanceFunc(target_point, dp);
        pq.push(ds);
    }
    return pq;
}

