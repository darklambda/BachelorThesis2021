#include <hip/hip_runtime.h>
#include "include/utils.cuh"
#include "../include/structs.h"
#include "../include/macros.h"

__device__ int IDX(int i, int j, int Lx, int* ex, int* ey){
	return i + ex[j] + ey[j] * Lx;
}

__device__ int IDXcm(int i, int j, int Lx, int Ly){
	return i + j * Lx * Ly;
}

void pointerSwap(cudaStruct *deviceOnly){
	prec *tempPtr = deviceOnly->f1;
	deviceOnly->f1 = deviceOnly->f2;
	deviceOnly->f2 = tempPtr;
}

void memoryFree(mainStruct host, mainStruct device, cudaStruct deviceOnly){
	delete[] host.b;
	delete[] host.w;

	hipFree(device.b);
	hipFree(device.w);
	
	hipFree(deviceOnly.h);
	hipFree(deviceOnly.f1);
	hipFree(deviceOnly.f2);
	hipFree(deviceOnly.binary1);
	hipFree(deviceOnly.binary2);
}

void memoryInit(configStruct config, cudaStruct *deviceOnly,
		 		mainStruct *device, mainStruct host){
	uint pBytes = config.Lx * config.Ly * sizeof(prec);
	//uint iBytes = config.Lx * config.Ly * sizeof(int);
	uint uBytes = config.Lx * config.Ly * sizeof(unsigned char);

	hipMalloc((void**)&(device->w), pBytes); 
	hipMalloc((void**)&(device->b), pBytes);

	hipMemcpy(device->w, host.w, pBytes, hipMemcpyHostToDevice);
	hipMemcpy(device->b, host.b, pBytes, hipMemcpyHostToDevice);

	hipMalloc((void**)&(deviceOnly->h), pBytes);
	hipMalloc((void**)&(deviceOnly->f1), 9 * pBytes);
	hipMalloc((void**)&(deviceOnly->f2), 9 * pBytes);
	hipMalloc((void**)&(deviceOnly->binary1), uBytes);
	hipMalloc((void**)&(deviceOnly->binary2), uBytes);
}

