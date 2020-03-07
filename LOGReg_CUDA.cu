#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


#define SZ 768
#define TOTITER 100000000
#define THRDS 1
//#define MAXITER TOTITER/THRDS

void checkCUDAError(const char* msg);

__host__ __device__ float Y_Model(float W1,float W2,float B,float X1,float X2){

	float Z = B + W1*X1 + W2*X2; 
	return 1.0/( 1.0 + exp(-Z));

}

__global__ void gd(float *X1,float *X2,float *Y,float *W1,float *W2,float *B){

	float h = 0.0001,y,dW1,dW2,dB;

    int idx;
    int MAXITER =  TOTITER/THRDS;
    unsigned int XX = 562628;
    unsigned int a = 1212*(threadIdx.x+1);
    unsigned int c = 3238 + (threadIdx.x+1);
    unsigned int m = 8191211;


    for(int i = 0; i<MAXITER; ++i)
    {
    	XX = (a*XX + c)%(m*(i+1)*(threadIdx.x+1)); //Linear Conguential Pseudo-Random Number Generator
        
        idx = XX%SZ;

        y = Y_Model(*W1,*W2,*B, X1[idx], X2[idx]);

	    dW1 = h*(Y[idx] - y)*y*(1.0 - y)*X1[idx];	    
	    dW2 = h*(Y[idx] - y)*y*(1.0 - y)*X2[idx];
		dB  =  h*(Y[idx] - y)*y*(1.0 - y);
         
        atomicAdd(W1, dW1);

        atomicAdd(W2, dW2);
         
        atomicAdd(B, dB);

    }

}

int main(){

    struct timeval start, end;
    srand (time(NULL));

	int numthreads=THRDS;
	int numblocks=1;
	
	float X[768][9];
	FILE *fp;


	float *h_X1;
	h_X1 = (float*)malloc(SZ*sizeof(float));
	float *h_X2;
	h_X2 = (float*)malloc(SZ*sizeof(float));

	float *h_Y;
	h_Y = (float*)malloc(SZ*sizeof(float));

	float *h_W1;
	h_W1 = (float*)malloc(sizeof(float));

	float *h_W2;
	h_W2 = (float*)malloc(sizeof(float));
	float *h_B;
	h_B = (float*)malloc(sizeof(float));

    *h_W1 = 0;
	*h_W2 = 0;
    *h_B = 0;
    
    float *d_X1, *d_X2, *d_Y, *d_W1, *d_W2, *d_B;

	cudaMalloc((void**)&d_X1,SZ*sizeof(float));

	cudaMalloc((void**)&d_X2,SZ*sizeof(float));

	cudaMalloc((void**)&d_Y,SZ*sizeof(float));

	cudaMalloc((void**)&d_W1,sizeof(float));

    cudaMalloc((void**)&d_W2,sizeof(float));

    cudaMalloc((void**)&d_B,sizeof(float));

	fp=fopen("input.txt","r");

	for(int i=0;i<SZ;i++){
		char *buff=(char*) malloc(70);

		fgets(buff, 70, fp);

		int count=0;

		int j=0;

		while(count<9){
			char *c=(char*) malloc(50);
			int l = 0;   
			while(buff[j]!=',' && buff[j]!='\0')
			{
				c[l] = buff[j];
				j++; l++; 

			}
			X[i][count] = atof(c);
			free (c);
			count++;
			if(count<9)
				j++;
		}

	}

	for(int i=0;i<SZ;i++)
	{
      h_X1[i] = X[i][2];
      h_X2[i] = X[i][5];
      h_Y[i] = X[i][8];
	}

	fclose(fp);

	cudaMemcpy(d_X1,h_X1,SZ*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_X2,h_X2,SZ*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_Y,h_Y,SZ*sizeof(float),cudaMemcpyHostToDevice);


	cudaMemcpy(d_W1,h_W1,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_W2,h_W2,sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_B,h_B,sizeof(float),cudaMemcpyHostToDevice);
	
    gettimeofday(&start,NULL);

	gd<<<numblocks,numthreads>>>(d_X1,d_X2, d_Y,d_W1,d_W2,d_B);
    cudaThreadSynchronize();

	checkCUDAError("kernel invocation");
	gettimeofday(&end,NULL);


    cudaMemcpy(h_W1, d_W1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W2, d_W2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, sizeof(float), cudaMemcpyDeviceToHost);

     float error = 0;

for(int i =0; i<SZ; ++i)
{
	error += pow((h_Y[i] - Y_Model(*h_W1,*h_W2,*h_B, h_X1[i], h_X2[i]) ),2);
}

error = sqrt(error);

error = error/SZ;

//int k = 10;
    std::cout<<"error "<<error<<'\n';
    
    printf("W1 = %f W2 = %f B = %f\n", *h_W1, *h_W2, *h_B);

    std::cout<<"Number of Threads: "<<numthreads<<'\n';
    std::cout<<"Total Number of Steps: "<<TOTITER<<'\n';
    std::cout<<"Time taken: \n"<<(end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec<<" microseconds. \n"; 

	return 0;
}

void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}
}

