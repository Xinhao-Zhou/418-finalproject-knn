#include "knn_header.h"
#include "knn_prep.cpp"

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

    for(DataPoint dp: data_train){
      //  printf("id: %d \n", dp.id);
    }

    //printf("test data\n");

    vector<DataPoint> data_test = parseFile_test(argc,argv);

    for(DataPoint dp: data_test){
        //printf("id: %d \n", dp.id);
    }

    int func = parseFunc(argv);

    vector<DataPoint> results = predictLables(data_test, data_train, 8, func);

    printf("\npredict results\n");

    int correctPrediction = 0;
    double accuracy = 0;
    for(int i = 0;i < results.size();i++){
        if(results[i].label == data_test[i].label){
            correctPrediction++;
        }
        //printf("test data point id: %d, predict label: %d, real label: %d\n", i, results[i].label, data_test[i].label);
    }

    accuracy = static_cast<double>(correctPrediction) / results.size();
    printf("accuracy: %lf\n", accuracy);
//    for(DataPoint dp : results){
//        printf("test data point id: %d, label : %d\n", dp.id, dp.label);
//    }

    return 0;
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
        //printf("char : %c, distance: %f\n", ds_char, ds_ds);
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
        //cout << iter->first << " : " << iter->second << endl;
        if(iter->second<temp_weight){
            temp_weight = iter->second;
            target_datapoint.label = iter->first;
        }
        iter++;
    }


    return target_datapoint;
}


vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
    vector<DataPoint> results;
    for(DataPoint test_dp : data_test){
        priority_queue<Distance> pq = getPriorityQueue(test_dp, data_train, func);

        vector<Distance> test;
        while(!pq.empty()){
            Distance ds = pq.top();
            //printf("priorityqueue src id :%d dst id: %d id dst label :%c distance :%d\n",ds.src_datapoint.id, ds.dest_datapoint.id, ds.dest_datapoint.label, ds.distance);
            pq.pop();
            test.push_back(ds);
        }

        for(Distance ds: test){
            pq.push(ds);
        }

        //printf("\nnearest neighbors\n");
        vector<Distance> nearestneighbors = findNeighbors(test_dp, pq, k);

        for(Distance ds: nearestneighbors){
            //printf("src id :%d dst id: %d id dst label :%c distance :%d\n", ds.src_datapoint.id, ds.dest_datapoint.id, ds.dest_datapoint.label, ds.distance);
        }

        //printf("\niterate map\n");
        test_dp = assignLabel(test_dp, nearestneighbors);

        //printf("\nassign label\n");
        //printf("test data point label: %c\n",test_dp.label);
        results.push_back(test_dp);
    }

    return results;
}

