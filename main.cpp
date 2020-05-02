
#include "knn_header.h"
#include "knn_prep.cpp"
#include "kmeans.h"
#include "cuda_kmeans.h"
#include "cuda_knn.h"
#include "cycletimer.h"
//#include "predict.cu"

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
    double *dataTest = getAttributesArray(data_test);
    int *labelTrain = getLabelArray(data_train);
    int *labelTest = getLabelArray(data_test);

    double seqStart = currentSeconds();
    clustersInit(data_train, 8);
    double seqEnd = currentSeconds();
    printf("sequential kmeans time: %lf\n",seqEnd - seqStart);
    cudaKmeans ckmeans = getClusters(dataTrain, data_train.size(), labelTrain, data_train[0].attributes.size(), 8);


    ckmeans = getClusterId(ckmeans, dataTest, data_test.size(),8, labelTest);    

//    double parKnnStart = currentSeconds();
    int correctPrediction = 0;

/*
    for(int i = 0;i < 8;i++){
	int *predictLabels = cuPredict(ckmeans.clusters[i].attributes, ckmeans.clusters[i].trainLabel,ckmeans.clusters[i].size,
		ckmeans.clusters[i].testAttr,ckmeans.clusters[i].testSize,data_train[0].attributes.size(), 16);
 
	for(int j = 0;j < ckmeans.clusters[i].testSize;j++){
		if(predictLabels[j] == ckmeans.clusters[i].testLabel[j])correctPrediction++;
	}  
    }
    double parKnnEnd = currentSeconds();
    double acc = (double)correctPrediction / (double)data_test.size();

    printf("parallel kmeans + knn time: %lf\n", parKnnEnd - parKnnStart);
    printf("correct: %d accuracy : %lf\n", correctPrediction, acc);
*/
    double parKnnStart = currentSeconds();

    cuPredict(dataTrain, labelTrain, data_train.size(), dataTest, data_test.size(), data_train[0].attributes.size(), 16);
    double parKnnEnd = currentSeconds();

    printf("parallel knn time: %lf\n", parKnnEnd - parKnnStart);
/*
    int *predictLabels = cuPredict(dataTrain, labelTrain, data_train.size(),
			dataTest, data_test.size(), data_train[0].attributes.size(), 16);

*/
/*
    printf("before kmeans");
    Kmeans kmeans = clustersInit(data_train, 8);

    printf("after kmeans");
    vector<DataPoint> results = predictLables(data_test, data_train, 8, func);

    printf("\npredict results\n");

    int correctPrediction = 0;
    double accuracy = 0;

    //int results_size = sizeof(results)/sizeof(DataPoint);
*/

/*
    int correctPrediction = 0;
    double accuracy = 0.f;
    for(int i = 0;i < data_test.size();i++){
            if(predictLabels[i] == data_test[i].label){
                correctPrediction++;
            }
else{
//		printf("predict :  %d actual: %d\n", predictLabels[i], data_test[i].label);	   
}
            //printf("test data point id: %d, predict label: %d, real label: %d\n", i, results[i].label, data_test[i].label);
        }

    accuracy = static_cast<double>(correctPrediction) / data_test.size();
    printf("accuracy: %lf\n", accuracy);
//    for(DataPoint dp : results){
//        printf("test data point id: %d, label : %d\n", dp.id, dp.label);
//    }
*/
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


// void assignLabel(DataPoint *target_datapoint, Distance *distances, int k){
//     //printf("assign labels");
// //    malloc and initialize

//     char *dis_map_keys = new char[k];
//     for(int i=0;i<k;i++){
//         dis_map_keys[i] = '#';
//     }
//     double *dis_map_values = new double[k];
//     for(int i=0; i<k;i++){
//         dis_map_values[i] = 0;
//     }

//     //assign values
//     for(int i=0;i<k;i++){
//     	//printf("%f\n", distances[i].distance);
//         char ds_char = distances[i].dest_datapoint.label;
//        // printf("label: %c\n", ds_char);
//         double ds_ds = distances[i].distance;
//         bool contained = false;
//         for(int j=0;j<i;j++){
//         	if(distances[j].dest_datapoint.label==ds_char){
//         		dis_map_values[j] += ds_ds;
//         		contained = true;
//         	}
//         }
//         if(!contained){
//         	dis_map_keys[i] = ds_char;
//         	dis_map_values[i] += ds_ds;
//         }

//     }

//     //get the smallest values
//     char final_label = dis_map_keys[0];
//     double final_distance = dis_map_values[0];

//     for(int i=0;i<k;i++){
//         if(dis_map_values[i]<final_distance && dis_map_keys[i]!='#'){
//             final_label = dis_map_keys[i];
//             final_distance = dis_map_values[i];
//         }
//     }

//     target_datapoint->label = final_label;
// }

int assignLabel(double *distances, int *ids, int *labels, int k){
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
        char ds_char = labels[i];
       // printf("label: %c\n", ds_char);
        double ds_ds = distances[i];
        bool contained = false;
        for(int j=0;j<i;j++){
            if(labels[j]==ds_char){
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

    return final_label;
    //target_datapoint->label = final_label;
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

double *getDistances(DataPoint dp, DataPoint *datapoints, int train_len, int func){
    double *distances = new double[train_len];
    for(int i=0;i<train_len;i++){
        distances[i] = distanceFunc(dp, datapoints[i], func);
    }
    return distances;
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

int *changeVectorToArray_ID(DataPoint *datapoints, int size){
    int *new_id = new int[size];
    for(int i=0;i<size;i++){
        new_id[i] = datapoints[i].id;
    }
    return new_id;

}

double *changeVectorToArray_Attributes(DataPoint *datapoints, int size, int attribute_len){
    double *ret = new double[size * attribute_len];
    for(int i = 0;i < size;i++){
        for(int j = 0;j < attribute_len;j++){
            ret[i * attribute_len + j] = datapoints[i].attributes[j];
        }
    }
    return ret;

}

int *changeVectorToArray_Labels(DataPoint *datapoints, int size){
    int *new_labels = new int[size];
    for(int i=0;i<size;i++){
        new_labels[i] = datapoints[i].label;
    }
    return new_labels;

}

int *getKthSmallestDatapoint_ID(int *id, int k){
    int *new_id = new int[k];
    for(int i=0;i<k;i++){
        new_id[i] = id[i];
    }

    return new_id;
}

double *getKthSmallestDatapoint_Attributes(double *attributes, int attribute_len, int k){
    double *new_attributes = new double[attribute_len*k];
    for(int i=0;i<k;i++){
        for(int j=0;j<attribute_len;j++){
            new_attributes[i*attribute_len+j] = attributes[i*attribute_len+j];
        }
    }
    return new_attributes;
}

// vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
 
//     vector<DataPoint> results;

//     int test_size = data_test.size();
//     int train_len = data_train.size();
//     int attribute_len = data_train[0].attributes.size();


//     DataPoint** sort_res = (DataPoint **)malloc(test_size*train_len*sizeof(DataPoint));
//     DataPoint** k_ret = (DataPoint **) malloc(test_size*train_len*sizeof(DataPoint));
//     Distance** k_dis = (Distance **) malloc(test_size*train_len*sizeof(Distance));

//     for(int count = 0;count<test_size;count++){
//     	DataPoint dp = data_test.at(count);
// 		sort_res[count] = sort_datapoint(dp, data_train, k, func);

//         double *distances = new double[train_len];
        
//         distances = getDistances(dp, sort_res[count], train_len, func);

// 	    int *new_id = new int[train_len];
//         double *new_attributes = new double[train_len * attribute_len];
        

//         new_id = changeVectorToArray_ID(sort_res[count], train_len);
//         new_attributes = changeVectorToArray_Attributes(sort_res[count], train_len, attribute_len);


//         k_ret[count] = getKthSmallestDatapoint(sort_res[count], k);

//         int *k_new_id = new int[k];
//         k_new_id = getKthSmallestDatapoint_ID(new_id, k);

//         double *k_new_attributes = new double[k * attribute_len];
//         k_new_attributes = getKthSmallestDatapoint_Attributes(new_attributes, attribute_len,k);

//         DataPoint dp = data_test.at(count);
//         k_dis[count] = getKthSmallestDistance(k_ret[count], dp, k, func);
//         assignLabel(&dp, k_dis[count], k);

//         results.push_back(dp);
        
//     }



//     free(sort_res);
//     free(k_ret);
//     free(k_dis);

//     return results;


// }





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

