#include "knn_header.h"
#include "cycletimer.h"
#include <omp.h>

#define OMP 1

vector<Distance> findNeighborsPQ(DataPoint target_point, priority_queue<Distance> distances, int k){
    vector<Distance> nearest_neighbors;

#if OMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(16)
#endif    	
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


#if OMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(16)
#endif   
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

//    int testId = 0;

    int *ret = new int[data_test.size()];
#if OMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(16)
#endif
    for(int testId = 0;testId < (int)data_test.size();testId++){

        DataPoint test_dp = data_test[testId];
        priority_queue<Distance> pq = getPriorityQueue(test_dp, data_train, func);
    
        vector<Distance> nearestneighbors = findNeighborsPQ(test_dp, pq, k);


        int prediction = assignLabelPQ(test_dp, nearestneighbors);

	ret[testId] = prediction;
   }


   for(int i = 0;i < (int)data_test.size();i++){
	results.push_back(ret[i]);
   }
   delete [] ret;
    return results;
}

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

int main(int argc, char *argv[]) {

    if (argc < 3) {
        printf("We need [train data set] [test data set] [distance function] [mode]\n");
        return -1;
    }

    vector <DataPoint> data_train = parseFile(argc, argv);

    vector <DataPoint> data_test = parseFile_test(argc, argv);


    int func = parseFunc(argv);

    printf("%s\n", "openmp implementation");
    double tmpStart = currentSeconds();
    vector<int> results = predictLablesPQ(data_test, data_train, 16, func);
    double tmpEnd = currentSeconds();
    int correctPrediction = 0;

    for(int i = 0;i < (int)data_test.size();i++){
        if(results[i] == data_test[i].label)correctPrediction++;
    }
    double acc = (double)correctPrediction / (double)data_test.size();
    printf("omp correct: %d accuracy: %lf\n", correctPrediction, acc);
    printf("omp time: %lf\n",tmpEnd-tmpStart);
}
