all:
	hipcc  -D IN=4 -D BN=3 -D PREC=64 src/cpp/files.cpp src/cu/LBM.cu src/cu/setup.cu src/main.cu -o bin/LBM
clean:
	rm bin/LBM
