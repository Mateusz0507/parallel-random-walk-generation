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

bool cuda_check_status(const char* file, const int line, bool terminate)
{
	cudaError_t error = cudaPeekAtLastError();
	return cuda_check(error, terminate);
}

void cuda_release(void** dev_ptr)
{
	if (dev_ptr && *dev_ptr)
	{
		cuda_check_terminate(cudaFree(*dev_ptr));
		dev_ptr = nullptr;
	}
}