#include "abstract_method.h"

algorithms::abstract_method::abstract_method()
	: releaser{cuda_memory_releaser::get()}
{
	
}

void algorithms::abstract_method::cuda_release(void** dev_ptr)
{
	if (dev_ptr && *dev_ptr)
	{
		releaser.unregister(*dev_ptr);
		cuda_check_terminate(cudaFree(*dev_ptr));
		*dev_ptr = nullptr;
	}
}

void algorithms::abstract_method::cuda_allocate(void** dev_ptr, int size, bool* allocation_failure)
{
	if (!*allocation_failure)
	{
		if (cuda_check_continue(cudaMalloc(dev_ptr, size)))
		{		
			releaser.log(*dev_ptr);
		}
		else
		{
			*allocation_failure = true;
			*dev_ptr = nullptr;
		}
	}
}
