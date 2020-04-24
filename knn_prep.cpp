#include "knn_header.h"

double distanceFunc(DataPoint datapoint1, DataPoint datapoint2, int func){
    double distance = 0.0f;
    int size = datapoint1.attributes.size();
    switch(func){
        case 0:{//Euclidean
            for(int i = 0;i < size;i++){
                double diff = datapoint1.attributes[i] - datapoint2.attributes[i];
                distance += pow(diff, 2);
            }
            distance = sqrt(distance);
            break;
        }
        case 1:{//Manhattan
            for(int i = 0;i < size;i++){
                double diff = fabs(datapoint1.attributes[i] - datapoint2.attributes[i]);
                distance += diff;
            }
            break;
        }
        case 2:{//Minkowski
            //assume p = 3
            int power = 3;
            for(int i = 0;i < size;i++){
                double diff = fabs(datapoint1.attributes[i] - datapoint2.attributes[i]);
                diff = pow(diff, power);
                distance += diff;
            }
            distance = pow(distance, 1 / power);
            break;
        }
    }

    return distance;
}

vector<DataPoint> _parseFile(char *FILE){
    char buf[256];
    vector<DataPoint> dataList;
    ifstream in(FILE);
    if(!in.is_open())exit(-1);

    //Get number of attributes
    //Peek the first line
    int len = in.tellg();
    int numberOfAttr = 0;

    in.getline(buf, 256);

    in.seekg(len, ios_base::beg);

    char deli[] = ",";
    char *temp = strtok(buf, deli);
    while(temp != NULL){
        temp = strtok(NULL, deli);
        numberOfAttr++;
    }

    vector<map<string, double> > discreteValuesDict(numberOfAttr);
    int instanceIdx = 0;
    while(!in.eof()){
        in.getline(buf, 256);

        char deli[] = ",";
        char *temp = strtok(buf, deli);
        int idx = 0;


        DataPoint newPoint;
        int newlineFlag = 0;
        while(temp != NULL){
            newlineFlag = 1;
//            if(idx == 0){
//                if(0 == sscanf(temp, "%d", &(newPoint.id))){
//                    // Not an integer
//                    if((discreteValuesDict.at(idx).find(temp)) == discreteValuesDict.at(idx).end()){
//                        //Not exists in the map
//                        int currentSize = discreteValuesDict.at(idx).size();
//                        discreteValuesDict.at(idx).insert(std::pair<string, double>(temp,
//                                static_cast<double>(currentSize)));
//                        newPoint.id = currentSize;
//                    }else{
//                        newPoint.id = static_cast<int>(discreteValuesDict.at(idx).at(temp));
//                    }
//                }
//            }else
            if(idx != numberOfAttr - 1){
                double attr;
                if(0 == sscanf(temp, "%lf", &attr)){
                    if((discreteValuesDict.at(idx).find(temp)) == discreteValuesDict.at(idx).end()) {
                        int currentSize = discreteValuesDict.at(idx).size();
                        discreteValuesDict.at(idx).insert(
                                std::pair<string, double>(temp, static_cast<double>(currentSize)));
                        newPoint.attributes.push_back(static_cast<double>(currentSize));
                    }else{
                        newPoint.attributes.push_back(discreteValuesDict.at(idx).at(temp));
                    }
                }else{
                    newPoint.attributes.push_back(attr);
                }
            }else{
                instanceIdx++;
                newPoint.id = instanceIdx;
                // Not an integer
                if((discreteValuesDict.at(idx).find(temp)) == discreteValuesDict.at(idx).end()){
                    //Not exists in the map
                    int currentSize = discreteValuesDict.at(idx).size();
                    discreteValuesDict.at(idx).insert(std::pair<string, double>(temp,
                                                                                static_cast<double>(currentSize)));
                    newPoint.label = currentSize;
                }else{
                    newPoint.label = static_cast<int>(discreteValuesDict.at(idx).at(temp));
                }
            }

            temp = strtok(NULL, deli);
            idx++;
        }
        if(newlineFlag) {
            dataList.push_back(newPoint);
        }
    }

    in.close();

    return dataList;
}

//shell> myprogram train_data_set test_data_set
vector<DataPoint> parseFile(int argc, char *argv[]){
    if(argc < 3){
        cout << "Error" <<endl;
        exit(-1);
    }
    vector<DataPoint> pq = _parseFile(argv[1]);

    return pq;
}

vector<DataPoint> parseFile_test(int argc, char *argv[]){
    vector<DataPoint> pq = _parseFile(argv[2]);

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

priority_queue<Distance> getPriorityQueue(DataPoint target_point, vector<DataPoint> datapoints, int func){
   priority_queue<Distance> pq;
   for(DataPoint dp:datapoints){
       Distance ds;
       ds.src_datapoint = target_point;
       ds.dest_datapoint = dp;
       ds.distance = distanceFunc(target_point, dp, func);
       pq.push(ds);
   }
   return pq;
}

