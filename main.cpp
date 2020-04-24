#include "knn_header.h"
#include "knn_prep.cpp"
#include "kmeans.cpp"

int parseFunc(char *argv[]){
    int funcNumber = 0;
    if(strcmp(argv[3], "euclidean") == 0){
        funcNumber = 0;
    }else if(strcmp(argv[3], "manhattan") == 0){
        funcNumber = 1;
    }else if(strcmp(argv[3], "minkowski") == 0){
        funcNumber = 2;
    }else{
        funcNumber = 0;
    }
    return funcNumber;
}

int main(int argc, char *argv[])
{
    cout << "Hello world!" << endl;

    printf("train data\n");

    if(argc < 4){
        printf("We need [train data set] [test data set] [distance function]\n");
        return -1;
    }

    vector<DataPoint> data_train = parseFile(argc, argv);

//    for(DataPoint dp: data_train){
//      //  printf("id: %d \n", dp.id);
//    }

    //printf("test data\n");

    vector<DataPoint> data_test = parseFile_test(argc,argv);

//    for(DataPoint dp: data_test){
//        //printf("id: %d \n", dp.id);
//    }

    int func = parseFunc(argv);

    Kmeans kmeans = clustersInit(data_train, 8);

    DataPoint *results = predictLables(data_test, data_train, 8, func);

    printf("\npredict results\n");

    int correctPrediction = 0;
    double accuracy = 0;

    int results_size = sizeof(results)/sizeof(DataPoint);
    for(int i = 0;i < results_size;i++){
            if(results[i].label == data_test[i].label){
                correctPrediction++;
            }
            //printf("test data point id: %d, predict label: %d, real label: %d\n", i, results[i].label, data_test[i].label);
        }

    accuracy = static_cast<double>(correctPrediction) / results_size;
    printf("accuracy: %lf\n", accuracy);
//    for(DataPoint dp : results){
//        printf("test data point id: %d, label : %d\n", dp.id, dp.label);
//    }

    return 0;
}




//vector<Distance> findNeighbors(DataPoint target_point, priority_queue<Distance> distances, int k){
//    vector<Distance> nearest_neighbors;
//
//    for(int i=0;i<k;i++){
//        Distance top_ds = distances.top();
//        nearest_neighbors.push_back(top_ds);
//        distances.pop();
//    }
//
//    return nearest_neighbors;
//
//}


DataPoint assignLabel(DataPoint target_datapoint, Distance *distances, int k){
//    malloc and initialize
    char *dis_map_keys = (char *)malloc(sizeof(char)*k);
    for(int i=0;i<k;i++){
        dis_map_keys[i] = '#';
    }
    double *dis_map_values = (double *)malloc(sizeof(double)*k);
    for(int i=0; i<k;i++){
        dis_map_values[i] = 0;
    }

    //assign values
    int distance_size = sizeof(distances)/sizeof(Distance);
    for(int i=0;i<distance_size;i++){
        char ds_char = distances[i].dest_datapoint.label;
        double ds_ds = distances[i].distance;
        dis_map_keys[i] = ds_char;
        dis_map_values[i] += ds_ds;
    }

    //get the smallest values
    char final_label = dis_map_keys[0];
    double final_distance = dis_map_values[0];

    for(int i=1;i<k;i++){
        if(dis_map_values[i]<final_distance && dis_map_keys[i]!='#'){
            final_label = dis_map_keys[i];
            final_distance = dis_map_values[i];
        }
    }

    target_datapoint.label = final_label;
     return target_datapoint;

//    map<char,double> dis_map;
//    for(Distance ds:distances){
//        char ds_char = ds.dest_datapoint.label;
//        double ds_ds = (double)ds.distance;
//        map<char,double>::iterator iter_char = dis_map.find(ds_char);
//        if(iter_char!=dis_map.end()){
//            ds_ds += dis_map.at(ds_char);
//            dis_map[ds_char] = ds_ds;
//        }else{
//            dis_map[ds_char] = ds_ds;
//        }
//    }
//
//    map<char,double>::iterator iter = dis_map.begin();
//    double temp_weight = (numeric_limits<double>::max)();
//    while(iter != dis_map.end()) {
//        //cout << iter->first << " : " << iter->second << endl;
//        if(iter->second<temp_weight){
//            temp_weight = iter->second;
//            target_datapoint.label = iter->first;
//        }
//        iter++;
//    }
//
//    return target_datapoint;

}


Distance *getSmallestDistances(DataPoint datapoint, DataPoint *data_train, int k, int func){
   // int data_set_size = data_train.length();
    int data_set_size = sizeof(data_train)/sizeof(DataPoint);
    DataPoint *temp_data_train = (DataPoint *)malloc(sizeof(DataPoint)*data_set_size);
    memcpy(temp_data_train, data_train, sizeof(DataPoint)*data_set_size);
    for(int i=0;i<data_set_size;i++){
        for(int j=0;j<data_set_size-i-1;j++){
            if(distanceFunc(datapoint, temp_data_train[j], func)>distanceFunc(datapoint, temp_data_train[j+1], func)){
                DataPoint temp = temp_data_train[j];
                temp_data_train[j] = temp_data_train[j+1];
                temp_data_train[j+1] = temp;
            }
        }
    }

    Distance *result = (Distance *) malloc(sizeof(DataPoint)*k);
    for(int i=0;i<k;i++){
        result[i].src_datapoint = datapoint;
        result[i].dest_datapoint = temp_data_train[i];
        result[i].distance = distanceFunc(datapoint, temp_data_train[i], func);
    }
//    memcpy(result, temp_data_train, sizeof(DataPoint)*k);

    free(temp_data_train);
    return result;

}


DataPoint *predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
//    vector<DataPoint> results;

    DataPoint * results = (DataPoint *)malloc(sizeof(DataPoint)*data_train.size());
    //change vector to array
    DataPoint *data_train_arr = (DataPoint *)malloc(sizeof(DataPoint)*data_train.size());
    for(int i=0;i<data_train.size();i++){
        data_train_arr[i] = data_train.at(i);
    }


    for(int i=0;i<data_test.size();i++){
        Distance *k_nearest_neighbors = getSmallestDistances(data_test[i], data_train_arr, k, func);
        data_test[i] = assignLabel(data_test[i], k_nearest_neighbors, k);
//        results.push_back(data_test[i]);
        results[i] = data_test[i];
    }


//    vector<DataPoint> results;
//    for(DataPoint test_dp : data_test){
//        priority_queue<Distance> pq = getPriorityQueue(test_dp, data_train, func);
//
//        vector<Distance> test;
//        vector<Distance> nearestneighbors = findNeighbors(test_dp, pq, k);
//        test_dp = assignLabel(test_dp, nearestneighbors, k);
//
//        results.push_back(test_dp);
//    }

    return results;
}

