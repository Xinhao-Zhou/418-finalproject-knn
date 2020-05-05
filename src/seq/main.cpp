
#include "knn_header.h"
#include "kmeans.h"
#include "cuda_kmeans.h"
#include "cuda_knn.h"
#include "cycletimer.h"
//#include "predict.cu"

vector<int> predictLablesPQ(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func);

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

int benchmark_mode(char *argv[]){
    int benchmark_mode = 0;
    if(strcmp(argv[4], "sequential") == 0){
        benchmark_mode = 0;
    }else if(strcmp(argv[4], "parallel") == 0){
        benchmark_mode = 1;
    }else{
        benchmark_mode = 1;
    }
    return benchmark_mode;
}


int main(int argc, char *argv[])
{
    printf("train data\n");

    if(argc <= 4){
        printf("We need [train data set] [test data set] [distance function] [mode] {-kmeans k} {-knn k}\n");
        return -1;
    }

    int Kmeans_K = 8;
    int Knn_K = 8;

    for(int i = 0;i < argc;i++){
        if(strcmp(argv[i], "-kmeans") == 0){
            sscanf(argv[i + 1], "%d", &Kmeans_K);
        }

        if(strcmp(argv[i], "-knn") == 0){
            sscanf(argv[i + 1], "%d", &Knn_K);
        }
    }


    printf("kmeans_k : %d, knn_k: %d\n",Kmeans_K, Knn_K);
    vector<DataPoint> data_train = parseFile(argc, argv);


    printf("test data\n");

    vector<DataPoint> data_test = parseFile_test(argc,argv);

    double *dataTrain = getAttributesArray(data_train);
    double *dataTest = getAttributesArray(data_test);
    int *labelTrain = getLabelArray(data_train);
    int *labelTest = getLabelArray(data_test);

    int mode = benchmark_mode(argv);
    
    if(mode == 1){
        double seqStart = currentSeconds();
        clustersInit(data_train, Kmeans_K);
        double seqEnd = currentSeconds();
        printf("sequential kmeans time: %lf\n",seqEnd - seqStart);


        double cudaStart = currentSeconds();
        cudaKmeans ckmeans = getClusters(dataTrain, data_train.size(), labelTrain, data_train[0].attributes.size(), Kmeans_K);
        double cudaEnd = currentSeconds();


        ckmeans = getClusterId(ckmeans, dataTest, data_test.size(), Kmeans_K, labelTest);    


        printf("cuda kmeans time: %lf\n", cudaEnd - cudaStart);

        double parKnnStart = currentSeconds();

        int correctPrediction = 0;

        int *predictLabel = cuPredictBasedOnKmeans(ckmeans, data_train.size(), 
            data_test.size(), data_train[0].attributes.size(), Knn_K, Kmeans_K);

        double parKnnEnd = currentSeconds();

        int aggrSize = 0;
        for(int i = 0;i < Kmeans_K;i++){
            for(int j = 0; j < ckmeans.clusters[i].testSize;j++){
                if(predictLabel[aggrSize + j] == ckmeans.clusters[i].testLabel[j]){
                    correctPrediction++;
                }
            }
            aggrSize += ckmeans.clusters[i].testSize;
        }

        delete [] predictLabel;
            
        double basicKnnStart = currentSeconds();
            predictLabel = cuPredict(dataTrain, labelTrain, data_train.size(), dataTest, data_test.size(), data_train[0].attributes.size(),
        	Knn_K);
        double basicKnnEnd = currentSeconds();




        double acc = (double)correctPrediction / (double)data_test.size();

        printf("cuda kmeans + knn time: %lf\n", parKnnEnd - parKnnStart);
        printf("cuda kmeans + knn correct: %d accuracy : %lf\n", correctPrediction, acc);
        printf("cuda knn time: %lf\n",basicKnnEnd - basicKnnStart);
    }

    if(mode==0){
        printf("%s\n", "baseline sequential");
        double tmpStart = currentSeconds();
        //vector<DataPoint> results = predictLables_seq(data_test, data_train, 16, func);
        vector<int> results = predictLablesPQ(data_test, data_train, Knn_K, 0);
        double tmpEnd = currentSeconds();
        int correctPrediction = 0;

        for(int i = 0;i < (int)data_test.size();i++){
            if(results[i] == labelTest[i])correctPrediction++;
        }
        double acc = (double)correctPrediction / (double)data_test.size();
        printf("baseline sequential correct: %d accuracy: %lf\n", correctPrediction, acc);
        printf("baseline sequential time: %lf\n",tmpEnd-tmpStart);
    }

    delete [] dataTrain;
    delete [] dataTest;
    delete [] labelTrain;
    delete [] labelTest;

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
        char ds_char = labels[i];
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
}

DataPoint *sort_datapoint(DataPoint target_datapoint, vector<DataPoint> data_train, int k, int func){
	
	int len = data_train.size();

    DataPoint *ret = new DataPoint[len];
    for(int i = 0;i < (int)data_train.size();i++){
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


vector<DataPoint> predictLables_seq(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
        vector<DataPoint> results;
    for(DataPoint test_dp : data_test){


     int len = data_train.size();

     DataPoint *ret = new DataPoint[len];

     ret = sort_datapoint(test_dp, data_train, k, func);

     DataPoint *k_ret = new DataPoint[k];

     k_ret = getKthSmallestDatapoint(ret, k);

     Distance *k_dis = new Distance[k];

     k_dis = getKthSmallestDistance(k_ret, test_dp, k, func);

     assignLabel_seq(&test_dp, k_dis, k);

         results.push_back(test_dp);
    }
    return results;


}

void assignLabel_seq(DataPoint *target_datapoint, Distance *distances, int k){


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
        char ds_char = distances[i].dest_datapoint.label;

        bool contained = false;
        for(int j=0;j<i;j++){
        	if(distances[j].dest_datapoint.label==ds_char){
        		dis_map_values[j] ++;
        		contained = true;
        	}
        }
        if(!contained){
        	dis_map_keys[i] = ds_char;
        	dis_map_values[i] += 1;
        }

    }

    //get the smallest values
    char final_label = dis_map_keys[0];
    double final_distance = dis_map_values[0];

    for(int i=0;i<k;i++){
        if(dis_map_values[i]>final_distance && dis_map_keys[i]!='#'){
            final_label = dis_map_keys[i];
            final_distance = dis_map_values[i];
        }
    }

    target_datapoint->label = final_label;
}



vector<Distance> findNeighborsPQ(DataPoint target_point, priority_queue<Distance> distances, int k){
    vector<Distance> nearest_neighbors;

    for(int i=0;i<k;i++){
        Distance top_ds = distances.top();
        nearest_neighbors.push_back(top_ds);
        distances.pop();
    }

    return nearest_neighbors;

}


int assignLabelPQ(DataPoint target_datapoint, vector<Distance> distances){
    vector<int> size;
    vector<int> label;

    for(int i = 0;i < (int)distances.size();i++){
	Distance tmp = distances[i];
	int tmpLabel = tmp.dest_datapoint.label;
        int containFlag = 0;
        int offset = 0;
	for(;offset < (int)label.size();offset++){
	    if(label[offset] == tmpLabel){
		containFlag = 1;
		break;
	    }
	}
	
	if(containFlag){
	    size[offset] += 1;
	}else{
	    label.push_back(tmpLabel);
	    size.push_back(1);
	}	
    }

    int maxSize = 0;
    int maxIndex = 0;
    for(int i = 0;i < (int)label.size();i++){
	if(size[i] > maxSize){
	    maxSize = size[i];
	    maxIndex = i;
	}	
    }

    return label[maxIndex];
}


vector<int> predictLablesPQ(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
    vector<int> results;
    for(DataPoint test_dp : data_test){
        priority_queue<Distance> pq = getPriorityQueue(test_dp, data_train, func);
	
        vector<Distance> test;

        vector<Distance> nearestneighbors = findNeighborsPQ(test_dp, pq, k);

        int prediction = assignLabelPQ(test_dp, nearestneighbors);
        results.push_back(prediction);
    }

    return results;
}
