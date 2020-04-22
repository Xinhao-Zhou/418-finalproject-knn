#include <iostream>
#include <vector>
#include <queue>
#include <map>
#include <limits>

using namespace std;

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
priority_queue<Distance> getPriorityQueue(DataPoint target_point, vector<DataPoint> datapoints);
vector<Distance> findNeighbors(DataPoint datapoint, priority_queue<Distance> train_datapoints, int k);
DataPoint assignLabel(DataPoint target_datapoint, vector<Distance> distances);
vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k);

int main()
{
    cout << "Hello world!" << endl;

    printf("train data\n");

    vector<DataPoint> data_train = parseFile();

    for(DataPoint dp: data_train){
        printf("id: %d \n", dp.id);
    }

    printf("test data\n");

    vector<DataPoint> data_test = parseFile_test();

    for(DataPoint dp: data_test){
        printf("id: %d \n", dp.id);
    }

    vector<DataPoint> results = predictLables(data_test, data_train, 2);

    printf("\npredict results\n");
    for(DataPoint dp : results){
        printf("test data point id: %d, label : %c", dp.id, dp.label);
    }

    return 0;
}

int distanceFunc(DataPoint datapoint1, DataPoint datapoint2){
    int N = rand() % 10+1;
    return N;
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



vector<Distance> findNeighbors(DataPoint target_point, priority_queue<Distance> distances, int k){
    vector<Distance> nearest_neighbors;

    for(int i=0;i<k;i++){
        Distance top_ds = distances.top();
        nearest_neighbors.push_back(top_ds);
        distances.pop();
    }

    return nearest_neighbors;

}


DataPoint assignLabel(DataPoint target_datapoint, vector<Distance> distances){
    map<char,double> dis_map;
    for(Distance ds:distances){
        char ds_char = ds.dest_datapoint.label;
        //double temp = (double)1/(ds.distance+1);
        //printf("temp : %f", temp);
        double ds_ds = (double)ds.distance;
        printf("char : %c, distance: %f\n", ds_char, ds_ds);
        map<char,double>::iterator iter_char = dis_map.find(ds_char);
        if(iter_char!=dis_map.end()){
            ds_ds += dis_map.at(ds_char);
            dis_map[ds_char] = ds_ds;
        }else{
            dis_map[ds_char] = ds_ds;
        }
    }

    map<char,double>::iterator iter = dis_map.begin();
    double temp_weight = (numeric_limits<double>::max)();
    while(iter != dis_map.end()) {
        cout << iter->first << " : " << iter->second << endl;
        if(iter->second<temp_weight){
            temp_weight = iter->second;
            target_datapoint.label = iter->first;
        }
        iter++;
    }


    return target_datapoint;
}


vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k){
    for(DataPoint test_dp : data_test){
        priority_queue<Distance> pq = getPriorityQueue(test_dp, data_train);

        vector<Distance> test;
        while(!pq.empty()){
            Distance ds = pq.top();
            printf("priorityqueue src id :%d dst id: %d id dst label :%c distance :%d\n",ds.src_datapoint.id, ds.dest_datapoint.id, ds.dest_datapoint.label, ds.distance);
            pq.pop();
            test.push_back(ds);
        }

        for(Distance ds: test){
            pq.push(ds);
        }

        printf("\nnearest neighbors\n");
        vector<Distance> nearestneighbors = findNeighbors(test_dp, pq, k);

        for(Distance ds: nearestneighbors){
            printf("src id :%d dst id: %d id dst label :%c distance :%d\n", ds.src_datapoint.id, ds.dest_datapoint.id, ds.dest_datapoint.label, ds.distance);
        }

        printf("\niterate map\n");
        test_dp = assignLabel(test_dp, nearestneighbors);

        printf("\nassign label\n");
        printf("test data point label: %c\n",test_dp.label);

    }

    return data_test;
}

