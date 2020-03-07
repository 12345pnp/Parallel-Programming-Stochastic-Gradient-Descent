#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//#define THRDS 1
#define MAXITER  100000000
#define SZ 768


double Y_model(double W1,double W2,double B,double X1,double X2)
{
  double Z = B + W1*X1 + W2*X2; 
  return 1.0/( 1.0 + exp(-Z));
}



int main(int argc, char *argv[]) {
struct timeval start, end;

char *str = new char[50];

double *X1 = new double[SZ];
double *X2 = new double[SZ];
double *X3 = new double[SZ];
double *X4 = new double[SZ];
double *X5 = new double[SZ];
double *X6 = new double[SZ];
double *X7 = new double[SZ];
double *X8 = new double[SZ];

double *Y = new double[SZ];

double W1 = 0, W2 = 0, B = 0;

double y;

double h = 0.0001;

std::ifstream inputD("input.txt");
char ch;
int idx,i = 0;

for(i = 0; i < SZ; i++)
{
 
  inputD>>X1[i]>>ch>>X2[i]>>ch>>X3[i]>>ch>>X4[i]>>ch>>X5[i]>>ch>>X6[i]>>ch>>X7[i]>>ch>>X8[i]>>ch>>Y[i];

}
srand (time(NULL));

gettimeofday(&start,NULL);

for(i = 0; i <MAXITER;++i)
{

idx = rand()%SZ;  
//h = 0.1*exp(-0.00001*i);
y = Y_model(W1,W2,B, X3[idx], X6[idx]);

W1 = W1 + h*(Y[idx] - y)*y*(1.0 - y)*X3[idx];
W2 = W2 +  h*(Y[idx] - y)*y*(1.0 - y)*X6[idx];
B = B +  h*(Y[idx] - y)*y*(1.0 - y);

}

gettimeofday(&end,NULL);

//int k = 739;

double error = 0;

for(i =0; i<SZ; ++i)
{
	error += pow((Y[i] - Y_model(W1,W2,B, X3[i], X6[i]) ),2);
}

error = sqrt(error);

error = error/SZ;

std::cout<<"error "<<error<<'\n';
std::cout<<"Coefficients "<<W1<<" "<<W2<<" "<<B<<'\n';
//std::cout<<Y_model(W1,W2,B,X3[k],X6[k])<<" "<<Y[k]<<'\n';

std::cout<<" \n Time taken: \n"<<(end.tv_sec - start.tv_sec)*1000000 + end.tv_usec - start.tv_usec<<" microseconds. \n"; 

   return 0;
 }
