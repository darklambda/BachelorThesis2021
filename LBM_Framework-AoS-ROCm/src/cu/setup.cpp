#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include "include/setup.cuh"
#include "include/utils.cuh"
#include "include/SWE.cuh"
#include "include/PDEfeq.cuh"
#include "../include/structs.h"
#include "../include/macros.h"

__global__ void binaryKernel(const configStruct config, 
	unsigned char* binary1, unsigned char* binary2) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < config.Lx*config.Ly) {
		unsigned char b1;
		unsigned char b2;
		int y = (int)i / config.Lx;
		int x = i - y * config.Lx;
		if (y == 0) {
			if (x == 0){
				b1 = 4 + 8 + 64;
				b2 = 1 + 2 + 16;
			}
			else if (x == config.Lx - 1){
				b1 = 1 + 8 + 128;
				b2 = 2 + 4 + 32;
			}
			else{
				b1 = 1 + 4 + 8 + 64 + 128;
				b2 = 2 + 16 + 32; 
			}
		}
		else if (y == config.Ly - 1) {
			if (x == 0) {
				b1 = 2 + 4 + 32;
				b2 = 1 + 8 + 128;
			}
			else if (x == config.Lx - 1){ 
				b1 = 1 + 2 + 16;
				b2 = 4 + 8 + 64;
			}
			else{ 
				b1 = 1 + 2 + 4 + 16 + 32;
				b2 = 8 + 64 + 128;
			}
		}
		else {
			if (x == 0){
				b1 = 2 + 4 + 8 + 32 + 64;
				b2 = 1 + 16 + 128;
			}
			else if (x == config.Lx - 1){
				b1 = 1 + 2 + 8 + 16 + 128;
				b2 = 4 + 32 + 64;
			}
			else{
				b1 = 255;
				b2 = 0;
			}
		}
		binary1[i] = b1;
		binary2[i] = b2;
	}
}

__device__ void calculateFeqSWE(prec* feq, prec* localMacroscopic, prec e){	
	prec factor = 1 / (9 * e*e);	
	prec localh = localMacroscopic[0];
	prec localux = localMacroscopic[1];
	prec localuy = localMacroscopic[2];
	prec gh  = 1.5 * 9.8 * localh;
	prec usq = 1.5 * (localux * localux + localuy * localuy);
	prec ux3 = 3.0 * e * localux;
	prec uy3 = 3.0 * e * localuy;
	prec uxuy5 = ux3 + uy3;
	prec uxuy6 = uy3 - ux3;

	feq[0] = localh * (1 - factor * (5.0 * gh + 4.0 * usq));
	feq[1] = localh * factor * (gh + ux3 + 4.5 * ux3*ux3 * factor - usq);
	feq[2] = localh * factor * (gh + uy3 + 4.5 * uy3*uy3 * factor - usq);
	feq[3] = localh * factor * (gh - ux3 + 4.5 * ux3*ux3 * factor - usq);
	feq[4] = localh * factor * (gh - uy3 + 4.5 * uy3*uy3 * factor - usq);
	feq[5] = localh * factor * 0.25 * (gh + uxuy5 + 4.5 * uxuy5*uxuy5 * factor - usq);
	feq[6] = localh * factor * 0.25 * (gh + uxuy6 + 4.5 * uxuy6*uxuy6 * factor - usq);
	feq[7] = localh * factor * 0.25 * (gh - uxuy5 + 4.5 * uxuy5*uxuy5 * factor - usq);
	feq[8] = localh * factor * 0.25 * (gh - uxuy6 + 4.5 * uxuy6*uxuy6 * factor - usq);
}

__global__ void fKernel(const configStruct config,
	const prec* __restrict__ h, prec* f) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	if (i < config.Lx*config.Ly) {
		prec feq[9];
		prec localMacroscopic[] = {h[i], 0, 0};
		#if PDE == 1
			calculateFeqSWE(feq, localMacroscopic, config.e);
		#elif PDE == 2
			calculateFeqHE(feq, localMacroscopic, config.e);
		#elif PDE == 3
			calculateFeqWE(feq, localMacroscopic, config.e);
		#elif PDE == 4
			calculateFeqNSE(feq, localMacroscopic, config.e);
		#elif PDE == 5
			calculateFeqUser(feq, localMacroscopic, config.e);
		#endif
		for (int j = 0; j < 9; j++)
			f[IDXcm(i, j)] = feq[j];
	}
}

__device__ void calculateFeqHE(prec* feq, prec* localMacroscopic, prec e){	
	prec factor = 1.0 / 9;	
	prec localT = localMacroscopic[0];

	feq[0] = localT * factor * 4;
	feq[1] = localT * factor;
	feq[2] = localT * factor;
	feq[3] = localT * factor;
	feq[4] = localT * factor;
	feq[5] = localT * factor * 0.25;
	feq[6] = localT * factor * 0.25;
	feq[7] = localT * factor * 0.25;
	feq[8] = localT * factor * 0.25;
}

__device__ int IDXcm(int i, int j){
	return 9*i + j;
}
