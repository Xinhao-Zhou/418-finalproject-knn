//
// Created by 郭三山 on 4/23/20.
//

#include "kmeans.h"
#include <stdlib.h>
#include <algorithm>

using namespace std;

void copyVector(DataPoint dp1, DataPoint dp2){
    dp1.id = dp2.id;
    dp1.label = dp2.label;
    for(int i = 0;i < dp2.attributes.size();i++){
        dp1.attributes.push_back(dp2.attributes[i]);
    }
}

Kmeans clustersInit(vector<DataPoint> points, int k){
    int iteration = 0;
    int totalSize = points.size();

    //generate random initial centers
    srand(k);
    vector<int> centralPoints;
    vector<Cluster> clusters;
    for(int i = 0; i < k; i++){
        int r = rand() % points.size();
        vector<int>::iterator res = find(centralPoints.begin(), centralPoints.end(), r);
        if(res == centralPoints.end()){
            //Not replicated
            //DataPoint *tmp = points.at(r);
            //DataPoint tmp;
            //copyVector(tmp, points[r]);
            Cluster tmpCluster;
            tmpCluster.size = 0;
            tmpCluster.centralPoint = points[r];
            clusters.push_back(tmpCluster);
            centralPoints.push_back(r);
        }else{
            i--;
        }
    }

    for(;iteration < MAX_ITERATION;iteration++){
        vector<vector<DataPoint> > nodeList(k);
        vector<DataPoint> newCenters;
        //Assign points to clusters according to central points
        for(int i = 0;i < totalSize;i++){
            double minDistance = INFINITY;
            int minIdx = -1;
            for(int j = 0;j < k;j++){
                double distance = distanceFunc(points[i], clusters[j].centralPoint, EUCLIDEAN);
                if(distance < minDistance){
                    minDistance = distance;
                    minIdx = j;
                }
            }
            nodeList[minIdx].push_back(points[i]);
        }


        //Get new central points
        for(int i = 0;i < k;i++){
            int attributesSize = points[0].attributes.size();
            double *centralPointAttributes = new double[attributesSize]();
            for(int j = 0;j < nodeList[i].size();j++){

                for(int l = 0;l < attributesSize;l++){
                    centralPointAttributes[l] += nodeList[i].at(j).attributes[l];
                }
            }

            DataPoint newCenterPoint;

            for(int j = 0;j < attributesSize;j++){
                double averageAttr = centralPointAttributes[j] / static_cast<double>(nodeList[i].size());
                newCenterPoint.attributes.push_back(averageAttr);
            }
            newCenters.push_back(newCenterPoint);
            delete [] centralPointAttributes;
        }
        
        //Compare new centers and old centers
        int quitFlag = 1;
        for(int i = 0; i < k;i++){
            double distance = distanceFunc(newCenters[i], clusters[i].centralPoint, EUCLIDEAN);
            if(distance > CONVERGE_THRESHOLD){
                quitFlag = 0;
            }
            clusters[i].centralPoint = newCenters[i];
        }

        if(quitFlag || iteration == MAX_ITERATION - 1){
            //Initialize clusters
            for(int i = 0;i < k;i++){
                for(int j = 0;j < nodeList[i].size();j++){
                    clusters[i].pointList.push_back(nodeList[i].at(j));
                }
                clusters[i].size = nodeList[i].size();
            }
            break;
        }
    }
    Kmeans ret(k);
    ret.clusters = clusters;

    return ret;
}

vector<DataPoint> getNearestClusterSet(DataPoint testInput, Kmeans clusters){
    double minDistance = INFINITY;
    int minIdx = -1;

    for(int i = 0;i < clusters.k; i++){
        double distance = distanceFunc(testInput, clusters.clusters[i].centralPoint, EUCLIDEAN);
        if(distance < minDistance){
            minDistance = distance;
            minIdx = i;
        }
    }

    return clusters.clusters[minIdx].pointList;
}

//int main(){
//    cout << "Hello World!" << endl;
//    return 0;
//}
