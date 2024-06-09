#include "kernels.cuh"

__global__ void kernel_improved_fitness_function(vector3* dev_random_walk, int N, 
	const real_t distance, const real_t precision, int* dev_invalid_distances, int* dev_random_walk_idx)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
	{
		int is_valid = 1;
		for (int i = 0; i < tid - 1; i++)
		{
			if (algorithms::model::get_distance(dev_random_walk[tid], dev_random_walk[i]) < distance - precision)
			{
				is_valid = 0;
				break;
			}
		}

		if (tid > 0 && 
			is_valid == 1 &&
			algorithms::model::get_distance(dev_random_walk[tid], dev_random_walk[tid - 1]) < distance - precision)
		{
			is_valid = 0;
		}

		dev_invalid_distances[tid] = is_valid;
		dev_random_walk_idx[tid] = tid;
	}
}


__global__ void kernel_init_tables(int generation_size, int* dev_generation_idx, int* dev_fitness_function)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < 2 * generation_size)
	{
		dev_generation_idx[tid] = tid;
		dev_fitness_function[tid] = INT_MAX;
	}
}