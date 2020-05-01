//
// Created by 郭三山 on 4/25/20.
//

#ifndef INC_418_FINALPROJECT_KNN_CUDA_KNN_H
#define INC_418_FINALPROJECT_KNN_CUDA_KNN_H


int *cuPredict(double *trainAttr, int* trainLabels, int trainSize,
        double *testAttr, int testSize, int attrSize, int k);
#endif //INC_418_FINALPROJECT_KNN_CUDA_KNN_H
