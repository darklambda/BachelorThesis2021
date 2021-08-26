#ifndef LBMKERNELS_CUH
	#define LBMKERNELS_CUH

	#include "../../include/structs.h"
	#include "../../include/macros.h"

	__global__ void First(const configStruct, prec*, prec*, prec*, const prec* __restrict__, const unsigned char* 
						    __restrict__, const unsigned char* __restrict__, const prec* __restrict__, 
						    prec*, prec*);
	__global__ void Second(const configStruct, prec*, prec*, prec*, const prec* __restrict__, const unsigned char* 
						    __restrict__, const unsigned char* __restrict__, const prec* __restrict__, 
						    prec*, prec*);
	__global__ void Third(const configStruct, prec*, prec*, prec*, const prec* __restrict__, const unsigned char* 
						    __restrict__, const unsigned char* __restrict__, const prec* __restrict__, 
						    prec*, prec*);

#endif
