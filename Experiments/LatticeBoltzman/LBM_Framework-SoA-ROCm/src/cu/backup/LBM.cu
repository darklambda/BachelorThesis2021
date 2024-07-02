#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include "include/setup.cuh"
#include "include/LBMkernels.cuh"
#include "include/SWE.cuh"
#include "include/utils.cuh"
#include "../cpp/include/files.h"
#include "../include/structs.h"
#include "../include/macros.h"
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void timeStep(configStruct config, mainStruct device, cudaStruct *deviceOnly, 
				 cudaEvent_t ct1, cudaEvent_t ct2, prec *msecs, double *times) {
	float dt;
	cudaEventRecord(ct1);

	prec* localf;
	uint arrayBytes = 9 * config.Lx * config.Ly * sizeof(prec);
	cudaMalloc((void**)&localf, arrayBytes); 

	prec* forcing;
	uint forcingBytes = 8 * config.Lx * config.Ly * sizeof(prec);
	cudaMalloc((void**)&forcing, forcingBytes); 

	prec* macro;
	uint macroBytes = 3 * config.Lx * config.Ly * sizeof(prec);
	cudaMalloc((void**)&macro, macroBytes); 

	double first_event = cpuSecond();

	First <<<config.gridSize,config.blockSize>>> (config, macro, forcing, localf, device.b, deviceOnly->binary1, 
								deviceOnly->binary2, deviceOnly->f1, deviceOnly->f2, deviceOnly->h);
	cudaDeviceSynchronize();
	double second_event = cpuSecond();
	cudaError_t err = cudaGetLastError();

     if ( err != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));       
     }
	Second <<<config.gridSize,config.blockSize>>> (config, macro, forcing, localf, device.b, deviceOnly->binary1, 
								deviceOnly->binary2, deviceOnly->f1, deviceOnly->f2, deviceOnly->h);
	
	cudaDeviceSynchronize();
	double third_event = cpuSecond();
	cudaError_t err2 = cudaGetLastError();

     if ( err2 != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err2));       
     }

	Third <<<config.gridSize,config.blockSize>>> (config, macro, forcing, localf, device.b, deviceOnly->binary1, 
								deviceOnly->binary2, deviceOnly->f1, deviceOnly->f2, deviceOnly->h);
	cudaDeviceSynchronize();
	double fourth_event = cpuSecond();
	cudaError_t err3 = cudaGetLastError();

     if ( err3 != cudaSuccess )
     {
        printf("CUDA Error: %s\n", cudaGetErrorString(err3));       
     }

	 times[0] += second_event - first_event;
	 times[1] += third_event - second_event;
	 times[2] += fourth_event - third_event;

	pointerSwap(deviceOnly);
	cudaFree(localf);
	cudaFree(forcing);
	cudaFree(macro);
	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);
	*msecs += dt;
}

void setup(configStruct config, mainStruct device, cudaStruct deviceOnly) {
	binaryKernel <<<config.gridSize,config.blockSize>>> (config, deviceOnly.binary1, deviceOnly.binary2);
	hKernel <<<config.gridSize,config.blockSize>>> (config, device.w, device.b, deviceOnly.h);
	fKernel <<<config.gridSize,config.blockSize>>> (config, deviceOnly.h, deviceOnly.f1);
}

void copyAndWriteResultData(configStruct config, mainStruct host, mainStruct device, cudaStruct deviceOnly, int t){
	wKernel <<<config.gridSize,config.blockSize>>> (config, deviceOnly.h, device.b, device.w);
	uint pBytes = config.Lx * config.Ly * sizeof(prec);
	cudaMemcpy(host.w, device.w, pBytes, cudaMemcpyDeviceToHost);
	writeOutput(config, t, host.w);
}

void LBM(configStruct config, mainStruct host, mainStruct device, cudaStruct *deviceOnly) {
	setup(config, device, *deviceOnly);

	int t = 0;
	cudaEvent_t ct1, ct2;
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	prec msecs = 0;

	double* times = new double[3]{ 0 };

	std::cerr << std::fixed << std::setprecision(1);
	while (t <= config.timeMax) {
		t++;
		timeStep(config, device, deviceOnly, ct1, ct2, &msecs, times);
		if (config.dtOut != 0 && t%config.dtOut == 0) {
			std::cout << "Time step: " << t << " (" << 100.0*t / config.timeMax << "%)" << std::endl;
			copyAndWriteResultData(config, host, device, *deviceOnly, t);
		}
	}

	printf("First Kernel: %f \nSecond Kernel: %f\nThird Kernel: %f\n", times[0], times[1], times[2]);
	delete[] times;

	if (config.dtOut == 0) 
		copyAndWriteResultData(config, host, device, *deviceOnly, t);
	std::cout << "Average time per time step: " << msecs / config.timeMax << "[ms]" << std::endl;
}

