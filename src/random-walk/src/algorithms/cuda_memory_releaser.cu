#include "cuda_memory_releaser.cuh"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"


bool cuda_memory_releaser::log(void* dev_ptr)
{
	auto& it = cuda_pointers.find(dev_ptr);
	if (it == cuda_pointers.end())
	{
		cuda_pointers.insert(dev_ptr);
	}
	return false;
}

bool cuda_memory_releaser::unregister(void* dev_ptr)
{
	auto& it = cuda_pointers.find(dev_ptr);
	if(it != cuda_pointers.end())
	{
		cuda_pointers.erase(dev_ptr);
		return true;
	}
	return false;
}

cuda_memory_releaser::~cuda_memory_releaser()
{
	for (auto& ptr : cuda_pointers)
	{
		cudaFree(ptr);
	}
}
