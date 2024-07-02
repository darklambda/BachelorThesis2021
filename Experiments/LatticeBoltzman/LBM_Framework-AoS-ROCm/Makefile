all:
	hipcc src/cpp/config.cpp src/cpp/input.cpp src/cpp/output.cpp src/cpp/utils.cpp src/cu/BC.cpp src/cu/SWE.cpp src/cu/utils.cpp src/cu/setup.cpp src/cu/PDEfeq.cpp src/cu/LBMkernels.cpp src/cu/LBM.cpp src/main.cpp -o bin/LBM

allv2:
	hipcc src/cpp/files.cpp src/cpp/config.cpp src/cpp/input.cpp src/cpp/output.cpp src/cpp/utils.cpp src/cu/BC.cpp src/cu/SWE.cpp src/cu/utils.cpp src/cu/setup.cpp src/cu/PDEfeq.cpp src/cu/LBMkernels.cpp src/cu/LBM.cpp src/main.cpp -o bin/LBM

clean:
	rm bin/LBM
