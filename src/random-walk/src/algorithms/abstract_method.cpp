#include "abstract_method.h"

void algorithms::abstract_method::cuda_release(void** dev_ptr)
{
	if (dev_ptr && *dev_ptr)
	{
		cuda_check_terminate(cudaFree(*dev_ptr));
		dev_ptr = nullptr;
	}
}
