#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "knn_header.h"

__global__ void get_smallest_datapoints_kernel(int k, int *id, int *new_id) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index<k){
		id[index] = new_id[index];
	}
}


__global__ void getKthSmallestDatapoint_Attributes_kernel(double *attributes, double *new_attributes, int attribute_len, int k){
	int index = blockIdx.x * blockDim.x + threadIdx.x;
    //double *new_attributes = new double[attribute_len*k];
    if(index<k){
    		for(int j=0;j<attribute_len;j++){
    				new_attributes[index*attribute_len+j] = attributes[index*attribute_len+j];
    		}
    }
   
}

vector<DataPoint> predictLables(vector<DataPoint> data_test, vector<DataPoint> data_train, int k, int func){
 
    vector<DataPoint> results;

    int test_size = data_test.size();
    int train_len = data_train.size();
    int attribute_len = data_train[0].attributes.size();

    const int threadsPerBlock = 512;
    const int blocks = (train_len + threadsPerBlock - 1) / threadsPerBlock;

    DataPoint** sort_res = (DataPoint **)malloc(test_size*train_len*sizeof(DataPoint));
    DataPoint** k_ret = (DataPoint **) malloc(test_size*train_len*sizeof(DataPoint));
    Distance** k_dis = (Distance **) malloc(test_size*train_len*sizeof(Distance));

    for(int count = 0;count<test_size;count++){
    	DataPoint dp = data_test.at(count);
		sort_res[count] = sort_datapoint(dp, data_train, k, func);

        double *distances = new double[train_len];
        
        distances = getDistances(dp, sort_res[count], train_len, func);

	    int *new_id = new int[train_len];
        double *new_attributes = new double[train_len * attribute_len];
        int *new_labels = new int[train_len];

        new_id = changeVectorToArray_ID(sort_res[count], train_len);
        new_attributes = changeVectorToArray_Attributes(sort_res[count], train_len, attribute_len);
        new_labels = changeVectorToArray_Labels(sort_res[count], train_len); 

       // for(int i=0;i<train_len;i++){
       // 	printf("%d : %d\n", sort_res[count][i].label, new_labels[i]);
       // }

        k_ret[count] = getKthSmallestDatapoint(sort_res[count], k);

        int *k_new_id = new int[k];

        int *kernel_id;
	    int *kernel_new_id;

	    cudaMalloc(&kernel_id, k*sizeof(int));
	    cudaMalloc(&kernel_new_id, k*sizeof(int));

	    cudaMemcpy(kernel_id, new_id, k*sizeof(int), cudaMemcpyHostToDevice);

	    get_smallest_datapoints_kernel<<<blocks, threadsPerBlock>>>(k, kernel_id, kernel_new_id);

	    cudaMemcpy(k_new_id, kernel_new_id, k*sizeof(int), cudaMemcpyDeviceToHost);
 
      //  k_new_id = getKthSmallestDatapoint_ID(new_id, k);

        double *k_new_attributes = new double[k * attribute_len];

        double *kernel_attributes;
        double *kernel_new_attributes;

        cudaMalloc(&kernel_attributes, k*attribute_len*sizeof(double));
        cudaMalloc(&kernel_new_attributes, k*attribute_len*sizeof(double));

        cudaMemcpy(kernel_attributes, new_attributes, k*attribute_len*sizeof(double), cudaMemcpyHostToDevice);

	    getKthSmallestDatapoint_Attributes_kernel<<<blocks, threadsPerBlock>>>(kernel_attributes, kernel_new_attributes, attribute_len, k);

	    cudaMemcpy(k_new_attributes, kernel_new_attributes, k*attribute_len*sizeof(double), cudaMemcpyDeviceToHost);

       // k_new_attributes = getKthSmallestDatapoint_Attributes(new_attributes, attribute_len,k);

       // DataPoint dp = data_test.at(count);

       // k_dis[count] = getKthSmallestDistance(k_ret[count], dp, k, func);
      
        //assignLabel(&dp, k_dis[count], k);
        int label = assignLabel(distances, new_id, new_labels, k);

        dp.label = label;

        results.push_back(dp);
        
    }



    free(sort_res);
    free(k_ret);
    free(k_dis);

    return results;


}

