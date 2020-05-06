# 418-finalproject-knn
## Files
### src: Source files including cuda and c++ files.<br>

>src/lib: Helper functions<br>
>src/cuda: Cuda kernels and interfaces of Kmeans and Knn<br>
>src/seq: Sequential version of Knn and Kmeans<br>
>src/omp: OpenMP version of Knn<br>
### data: Test data in CSV format. You can use the test data to run Kmeans and Knn.  
 >Car: A dataset from UCI machine learning repository. It has small size and is good for validation.<br>
 >Adult: A dataset from UCI machine learning repository. It has relatively large size.<br>
 >NomAdult: Normalized Adult dataset.<br>


## Compile
We test the code on GHC cluster. NVCC and g++ are required to compile. Just use `make` to get the sequential and CUDA version of Knn and Kmeans. Use `make ompknn` to get openmp version of Knn.<br>

## Running
>To run, just use<br>
>>knn /path/to/trainset /path/to/testset  (enclidean|manhattan|minkowski)  (sequential | parallel) [-kmeans n] [-knn n]<br>

>>The first 3 parameters are necessary while the other 3 are optional.
