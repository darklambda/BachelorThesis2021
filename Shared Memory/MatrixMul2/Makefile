
nvcc: clean
	@echo 'Compiling CUDA...'
	@nvcc -Wno-deprecated-gpu-targets -gencode arch=compute_50,code=sm_50 -o matrixMulCUDA matrixMul.cu

hipify: clean
	@echo 'Compiling ROCm...'
	@hipify-perl matrixMul.cu > matrixMul.cpp
	@sed -i '171s/CUDA/ROCm/' matrixMul.cpp
	@hipcc -arch=compute_50 -code=sm_50 -o matrixMulHIP matrixMul.cpp

hip: clean
	@echo 'Compiling ROCm...'
	@hipify-perl matrixMul.cu > matrixMul.cpp
	@sed -i '171s/CUDA/ROCm/' matrixMul.cpp
	@sed '35d' matrixMul.cpp
	@hipcc -o matrixMulHIP matrixMul.cpp

executeCudaIter: nvcc hipify
	for number in 1 ; do \
		echo $$number ; \
		./matrixMulCUDA 3200 320 320 3200 $$number; \
		./matrixMulHIP 3200 320 320 3200 $$number; \
	done

executeCudaSize: nvcc hipify
	mkdir output
	touch /output/computeTimeHIP-size.txt
	touch /output/computeTimeCUDA-size.txt
	for number in 320 640 960 3200 6400 9600  ; do \
		echo $$number ; \
		./matrixMulCUDA $$number $$number $$number $$number 10 >> computeTimeCUDA-size.txt ; \
		./matrixMulHIP $$number $$number $$number $$number 10 >> computeTimeHIP-size.txt ; \
	done

executeHipIter: hip
	mkdir output
	touch /output/computeTimeHIP.txt
	for number in 1 10 100 1000 10000 100000 1000000 ; do \
		echo $$number ; \
		./matrixMulHIP 3200 320 320 3200 $$number >> computeTimeHIP.txt ; \
	done

executeHipSize: hip
	mkdir output
	touch /output/computeTimeHIP-size.txt
	for number in 320 640 960 3200 6400 9600 ; do \
		echo $$number ; \
		./matrixMulHIP $$number $$number $$number $$number 10 >> computeTimeHIP-size.txt ; \
	done

clean:
	rm -f matrixMulHIP matrixMulCUDA matrixMul.cpp