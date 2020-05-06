# 418-finalproject-knn
## Files
### src: Source files including cuda and c++ files.<br>

>src/lib: Helper functions<br>
>src/cuda: Cuda kernels and interfaces of Kmeans and Knn<br>
>src/seq: Sequential version of Knn and Kmeans<br>
>src/omp: OpenMP version of Knn<br>
### data: Test data in CSV format. You can use the test data to run Kmeans and Knn.  
 >Car: A dataset from UCI machine learning repository. It has small size and is good for validation.
 >Adult: A dataset from UCI machine learning repository. It has relatively large size.
 >NomAdult: Normalized Adult dataset.



To run the knn, just use:
>>knn car_train.data car_test.data [ enclidean | manhattan | minkowski ] [sequential | parallel]
