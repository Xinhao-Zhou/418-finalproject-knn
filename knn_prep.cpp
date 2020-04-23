#include "..\knn_header.h"

int distanceFunc(DataPoint datapoint1, DataPoint datapoint2){
    int N = rand() % 10+1;
    return N;
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

    while(!in.eof()){
        in.getline(buf, 256);

        char deli[] = ",";
        char *temp = strtok(buf, deli);
        int idx = 0;


        DataPoint newPoint;
        newPoint.id = -1;
        while(temp != NULL){
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
        if(newPoint.id != -1) {
            dataList.push_back(newPoint);
        }
    }

    in.close();

    return dataList;
}

vector<DataPoint> parseFile(int argc, char *argv[]){
    if(argc < 2){
        cout << "Error" <<endl;
        exit(-1);
    }
    vector<DataPoint> pq = _parseFile(argc, argv);

    return pq;
}

vector<DataPoint> parseFile_test(int argc, char *argv[]){
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

