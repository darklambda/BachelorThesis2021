all:
	g++ -c -O3 -Wall -I. -I.. -I../../common/inc matrixMul_gold.cpp -o matrixMul_gold.o
	hipcc -Wno-deprecated-gpu-targets -I$ROCM_PATH -I. -I.. -I../../common/inc -c matrixMul.cpp -o matrixMul.o

