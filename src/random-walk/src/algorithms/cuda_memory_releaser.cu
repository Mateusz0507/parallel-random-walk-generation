#include "cuda_memory_releaser.cuh"
#include "cuda_device_runtime_api.h"
#include "device_launch_parameters.h"
#include "common/common.cuh"

cuda_memory_releaser cuda_memory_releaser::releaser;

cuda_memory_releaser& cuda_memory_releaser::get()
{
	return releaser;
}

bool cuda_memory_releaser::log(void* dev_ptr)
{
	auto it = cuda_pointers.find(dev_ptr);
	if (it == cuda_pointers.end())
	{
		cuda_pointers.insert(dev_ptr);
		return true;
	}
	return false;
}

bool cuda_memory_releaser::unregister(void* dev_ptr)
{
	auto it = cuda_pointers.find(dev_ptr);
	if(it != cuda_pointers.end())
	{
		cuda_pointers.erase(dev_ptr);
		return true;
	}
	return false;
}

cuda_memory_releaser::~cuda_memory_releaser()
{
	for (auto& dev_ptr : cuda_pointers)
	{
		cuda_check_continue(cudaFree(dev_ptr));
	}
}
