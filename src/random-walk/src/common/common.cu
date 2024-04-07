#include "common/common.cuh"


bool cuda_check_error(cudaError_t error, const char* file, const int line, bool terminate)
{
	if (error != cudaSuccess)
	{
		std::cerr << "CUDA error: " << cudaGetErrorString(error) << ",  occurred in file: "
			<< file << ", line: " << line << "." << std::endl;
		if (terminate)
			exit(EXIT_FAILURE);
		return false;
	}
	return true;
}
