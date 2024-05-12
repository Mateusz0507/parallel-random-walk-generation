#include "abstract_method.h"

void algorithms::abstract_method::cuda_release(void** dev_ptr)
{
	if (dev_ptr && *dev_ptr)
	{
		cuda_check_terminate(cudaFree(*dev_ptr));
		dev_ptr = nullptr;
	}
}

void algorithms::abstract_method::cuda_allocate(void** dev_ptr, int size, bool* allocation_failure)
{
	if (!*allocation_failure && !cuda_check_continue(cudaMalloc(dev_ptr, size)))
	{
		*allocation_failure = true;
		*dev_ptr = nullptr;
	}
}
