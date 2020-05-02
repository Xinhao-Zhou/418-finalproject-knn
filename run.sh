#!/bin/bash

make clean
make
./knn nomadult_train.csv nomadult_test.csv en
