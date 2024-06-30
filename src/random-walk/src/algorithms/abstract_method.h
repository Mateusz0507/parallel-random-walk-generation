#pragma once
#include "algorithms/model/particle.cuh"
#include "algorithms/cuda_memory_releaser.cuh"

namespace algorithms
{
	class abstract_method
	{
	public:
		abstract_method();

		virtual bool run(vector3** result, void*) = 0;
	protected:
		cuda_memory_releaser& releaser;

		void cuda_release(void** dev_ptr);
		void cuda_allocate(void** dev_ptr, int size, bool* allocation_failure);
	};
}
