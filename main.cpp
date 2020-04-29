#include "knn_header.h"
#include "knn_prep.cpp"
#include "kmeans.cpp"
#include "cuda_kmeans.h"
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

    printf("test data\n");

    vector<DataPoint> data_test = parseFile_test(argc,argv);

//    for(DataPoint dp: data_test){
//        //printf("id: %d \n", dp.id);
//    }

    int func = parseFunc(argv);

    double *dataTrain = getAttributesArray(data_train);

    vector<DataPoint> results = predictLables(data_test, data_train, 8, func);

    printf("\npredict results\n");

    int correctPrediction = 0;
    double accuracy = 0;

    //int results_size = sizeof(results)/sizeof(DataPoint);
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


void assignLabel(DataPoint *target_datapoint, Distance *distances, int k){
    //printf("assign labels");
//    malloc and initialize

    char *dis_map_keys = new char[k];
    for(int i=0;i<k;i++){
        dis_map_keys[i] = '#';
    }
    double *dis_map_values = new double[k];
    for(int i=0; i<k;i++){
        dis_map_values[i] = 0;
    }

    //assign values
    for(int i=0;i<k;i++){
    	//printf("%f\n", distances[i].distance);
        char ds_char = distances[i].dest_datapoint.label;
       // printf("label: %c\n", ds_char);
        double ds_ds = distances[i].distance;
        bool contained = false;
        for(int j=0;j<i;j++){
        	if(distances[j].dest_datapoint.label==ds_char){
        		dis_map_values[j] += ds_ds;
        		contained = true;
        	}
        }
        if(!contained){
        	dis_map_keys[i] = ds_char;
        	dis_map_values[i] += ds_ds;
        }

    }

    //get the smallest values
    char final_label = dis_map_keys[0];
    double final_distance = dis_map_values[0];

    for(int i=0;i<k;i++){
        if(dis_map_values[i]<final_distance && dis_map_keys[i]!='#'){
            final_label = dis_map_keys[i];
            final_distance = dis_map_values[i];
        }
    }

    target_datapoint->label = final_label;
//s     return target_datapoint;

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



DataPoint *sort_datapoint(DataPoint target_datapoint, vector<DataPoint> data_train, int k, int func){
	
	int len = data_train.size();

    DataPoint *ret = new DataPoint[len];
    for(int i = 0;i < data_train.size();i++){
        ret[i] = data_train.at(i);
    }

    for(int i=0;i<len;i++){
        for(int j=0;j<len-i-1;j++){
            if(distanceFunc(target_datapoint, ret[j], func)>distanceFunc(target_datapoint, ret[j+1], func)){
                DataPoint temp = ret[j];
                ret[j] = ret[j+1];
                ret[j+1] = temp;
            }
        }
    }

    return ret;
}

DataPoint *getKthSmallestDatapoint(DataPoint *datapoints, int k){
	DataPoint *k_ret = new DataPoint[k];
	for(int i=0;i<k;i++){
		k_ret[i] = datapoints[i];
	}

	return k_ret;
}

Distance *getKthSmallestDistance(DataPoint *k_smallest, DataPoint target_datapoint, int k, int func){
	Distance *k_dis = new Distance[k];
	for(int i=0;i<k;i++){
		Distance dis;
		dis.src_datapoint = target_datapoint;
		dis.dest_datapoint = k_smallest[i];
		dis.distance = distanceFunc(k_smallest[i], target_datapoint, func);
		k_dis[i] = dis;
	}
	return k_dis;
}


vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
 
    vector<DataPoint> results;

    int test_size = data_test.size();
    int train_len = data_train.size();

    //DataPoint sort_res[test_size][train_len];

    DataPoint** sort_res = (DataPoint **)malloc(test_size*train_len*sizeof(DataPoint));
    DataPoint** k_ret = (DataPoint **) malloc(test_size*train_len*sizeof(DataPoint));
    Distance** k_dis = (Distance **) malloc(test_size*train_len*sizeof(Distance));

    for(int count = 0;count<test_size;count++){
    	DataPoint dp = data_test.at(count);
		sort_res[count] = sort_datapoint(dp, data_train, k, func);
		k_ret[count] = getKthSmallestDatapoint(sort_res[count], k);
		k_dis[count] = getKthSmallestDistance(k_ret[count], dp, k, func);
		assignLabel(&dp, k_dis[count], k);

       	results.push_back(dp);
    }

    return results;


}


// vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
 
//     vector<DataPoint> results;
//     for(DataPoint test_dp : data_test){


//     	int len = data_train.size();

// 	    DataPoint *ret = new DataPoint[len];

// 		ret = sort_datapoint(test_dp, data_train, k, func);

// 		DataPoint *k_ret = new DataPoint[k];

// 		k_ret = getKthSmallestDatapoint(ret, k);

// 		Distance *k_dis = new Distance[k];

// 		k_dis = getKthSmallestDistance(k_ret, test_dp, k, func);

// 		assignLabel(&test_dp, k_dis, k);

//        	results.push_back(test_dp);

       

//     	// priority_queue<Distance> pq = getPriorityQueue(test_dp, data_train, func);
//      //    vector<Distance> test;
//      //    vector<Distance> nearestneighbors = findNeighbors(test_dp, pq, k);
//      //     Distance *nearestneighbors_arr = (Distance *)malloc(sizeof(Distance)*k);
// 	    // for(int i=0;i<nearestneighbors.size();i++){
// 	    // 	//printf("%f\n", nearestneighbors.at(i).distance);
// 	    //     memcpy(&nearestneighbors_arr[i], &nearestneighbors.at(i), sizeof(Distance));
// 	    // }

//     	// assignLabel(&test_dp, nearestneighbors_arr, k);

//      //   results.push_back(test_dp);
//     }

//     return results;


// }

