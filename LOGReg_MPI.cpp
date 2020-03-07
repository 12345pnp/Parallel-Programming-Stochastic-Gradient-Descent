#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <math.h>
#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAXITER  100000000
#define SZ 768


double Y_model(double W1,double W2,double B,double X1,double X2)
{
	double Z = B + W1*X1 + W2*X2; 
	return 1.0/( 1.0 + exp(-Z));
}


int main(int argc, char *argv[]) {
    double start, end;
	int rank, size;
    MPI_Comm comm = MPI_COMM_WORLD;
     
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

      
	char *str = new char[50];

	int chunk = MAXITER/size;
    
    int sz_small = SZ/size;

	double *X1 = new double[SZ];
	double *X2 = new double[SZ];
	double *X3 = new double[SZ];
	double *X4 = new double[SZ];
	double *X5 = new double[SZ];
	double *X6 = new double[SZ];
	double *X7 = new double[SZ];
	double *X8 = new double[SZ];

	double *Y = new double[SZ];

	double WSUM1=0, WSUM2 = 0, BSUM = 0, W1 = 0, W2 = 0, B = 0, dW1 = 0, dW2 = 0, dB = 0;

	double y;


	double h = 0.0001;

	std::ifstream inputD("input.txt");
	char ch;
	int idx,i = 0;

	for(i = 0; i < SZ; i++)
	{

		inputD>>X1[i]>>ch>>X2[i]>>ch>>X3[i]>>ch>>X4[i]>>ch>>X5[i]>>ch>>X6[i]>>ch>>X7[i]>>ch>>X8[i]>>ch>>Y[i];

	}

start = MPI_Wtime(); 


		for(i = 0; i <chunk;++i)
		{

			idx = rand()%sz_small + rank*sz_small;  
			y = Y_model(W1,W2,B, X3[idx], X6[idx]);

			dW1 = h*(Y[idx] - y)*y*(1.0 - y)*X3[idx];
			dW2 = h*(Y[idx] - y)*y*(1.0 - y)*X6[idx];
			dB  =  h*(Y[idx] - y)*y*(1.0 - y);

			W1 = W1 + dW1;

			W2 = W2 + dW2;

			B =  B +   dB;

		}

MPI_Barrier(comm);

MPI_Reduce(&W1, &WSUM1, 1, MPI_DOUBLE,MPI_SUM, 0, comm);
               

MPI_Reduce(&W2, &WSUM2, 1, MPI_DOUBLE,MPI_SUM, 0, comm);
               
MPI_Reduce(&B, &BSUM, 1, MPI_DOUBLE,MPI_SUM, 0, comm);
               


MPI_Barrier(comm);

MPI_Finalize();

if(rank>0)
 return 0;

end = MPI_Wtime(); 

W1 = WSUM1/size;
W2 = WSUM2/size;
B =  BSUM/size;

double error = 0;

for(i =0; i<SZ; ++i)
{
	error += pow((Y[i] - Y_model(W1,W2,B, X3[i], X6[i]) ),2);
}

error = sqrt(error);

error = error/SZ;

//int k = 10;
std::cout<<"error "<<error<<'\n';

printf("Coefficients: %lf %lf %lf \n ", W1, W2, B);
//printf(" %lf %lf \n ", Y_model(W1,W2,B,X3[k],X6[k]),Y[k]);

printf("Time taken (using %d cores): %g microseconds. \n\n", size, (end - start)*1000000);



	return 0;
}
