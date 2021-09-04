#include "hip/hip_runtime.h"
#include <iostream>
#include <iomanip>
#include <hip/hip_runtime.h>
#include "include/setup.cuh"
#include "include/LBMkernels.cuh"
#include "include/SWE.cuh"
#include "include/utils.cuh"
#include "../cpp/include/input.h"
#include "../cpp/include/output.h"
#include "../cpp/include/config.h"
#include "../include/structs.h"
#include "../include/macros.h"
#include <sys/time.h>

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void timeStep(configStruct config, mainStruct device, cudaStruct *deviceOnly, 
				 hipEvent_t ct1, hipEvent_t ct2, prec *msecs, double *times) {
	float dt;
	hipEventRecord(ct1);

	prec* localf;
	uint arrayBytes = 9 * config.Lx * config.Ly * sizeof(prec);
	hipMalloc((void**)&localf, arrayBytes); 

	prec* forcing;
	uint forcingBytes = 8 * config.Lx * config.Ly * sizeof(prec);
	hipMalloc((void**)&forcing, forcingBytes); 

	prec* macro;
	uint macroBytes = 3 * config.Lx * config.Ly * sizeof(prec);
	hipMalloc((void**)&macro, macroBytes); 

	double first_event = cpuSecond();

	hipLaunchKernelGGL(First, dim3(config.gridSize), dim3(config.blockSize), 0, 0, config, macro, forcing, localf, device.b, deviceOnly->binary1, 
								deviceOnly->binary2, deviceOnly->f1, deviceOnly->f2, deviceOnly->h);
	hipDeviceSynchronize();
	double second_event = cpuSecond();
	hipError_t err = hipGetLastError();

     if ( err != hipSuccess )
     {
        printf("CUDA Error: %s\n", hipGetErrorString(err));       
     }
	hipLaunchKernelGGL(Second, dim3(config.gridSize), dim3(config.blockSize), 0, 0, config, macro, forcing, localf, device.b, deviceOnly->binary1, 
								deviceOnly->binary2, deviceOnly->f1, deviceOnly->f2, deviceOnly->h);
	
	hipDeviceSynchronize();
	double third_event = cpuSecond();
	hipError_t err2 = hipGetLastError();

     if ( err2 != hipSuccess )
     {
        printf("CUDA Error: %s\n", hipGetErrorString(err2));       
     }

	hipLaunchKernelGGL(Third, dim3(config.gridSize), dim3(config.blockSize), 0, 0, config, macro, forcing, localf, device.b, deviceOnly->binary1, 
								deviceOnly->binary2, deviceOnly->f1, deviceOnly->f2, deviceOnly->h);
	hipDeviceSynchronize();
	double fourth_event = cpuSecond();
	hipError_t err3 = hipGetLastError();

     if ( err3 != hipSuccess )
     {
        printf("CUDA Error: %s\n", hipGetErrorString(err3));       
     }

	 times[0] += second_event - first_event;
	 times[1] += third_event - second_event;
	 times[2] += fourth_event - third_event;

	pointerSwap(deviceOnly);
	hipFree(localf);
	hipFree(forcing);
	hipFree(macro);
	hipEventRecord(ct2);
	hipEventSynchronize(ct2);
	hipEventElapsedTime(&dt, ct1, ct2);
	*msecs += dt;
}

void setup(configStruct config, mainStruct device, cudaStruct deviceOnly) {
	hipLaunchKernelGGL(binaryKernel, dim3(config.gridSize), dim3(config.blockSize), 0, 0, config, deviceOnly.binary1, deviceOnly.binary2);
	hipLaunchKernelGGL(hKernel, dim3(config.gridSize), dim3(config.blockSize), 0, 0, config, device.w, device.b, deviceOnly.h);
	hipLaunchKernelGGL(fKernel, dim3(config.gridSize), dim3(config.blockSize), 0, 0, config, deviceOnly.h, deviceOnly.f1);
}

void copyAndWriteResultData(configStruct config, mainStruct host, mainStruct device, cudaStruct deviceOnly, int t){
	hipLaunchKernelGGL(wKernel, dim3(config.gridSize), dim3(config.blockSize), 0, 0, config, deviceOnly.h, device.b, device.w);
	uint pBytes = config.Lx * config.Ly * sizeof(prec);
	hipMemcpy(host.w, device.w, pBytes, hipMemcpyDeviceToHost);
	writeOutput(config, t, host.w);
}

void LBM(configStruct config, mainStruct host, mainStruct device, cudaStruct *deviceOnly) {
	setup(config, device, *deviceOnly);

	int t = 0;
	hipEvent_t ct1, ct2;
	hipEventCreate(&ct1);
	hipEventCreate(&ct2);
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

