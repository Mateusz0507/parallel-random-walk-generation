#pragma once
#include "algorithms/model/randomization.cuh"

__global__ void algorithms::randomization::kernel_setup(curandState* dev_states, int N, uint64_t seed, uint64_t offset = 0)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N) curand_init(seed, tid, offset, &dev_states[tid]);
}

__global__ void algorithms::randomization::kernel_generate_random_unit_vectors(vector3* dev_unit_vectors, curandState* dev_states, int N)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
	{
		generate_random_unit_vector(&dev_unit_vectors[tid], &dev_states[tid]);
	}
}

void algorithms::randomization::generate_random_unit_vector(vector3* dev_dst_unit_vector, curandState* dev_state)
{
	real_t alpha = acos(2 * cuda_rand_uniform(dev_state) - 1.0);
	real_t beta = cuda_rand_uniform(dev_state) * 2 * PI;
	dev_dst_unit_vector->x = sin(alpha) * cos(beta);
	dev_dst_unit_vector->y = sin(alpha) * sin(beta);
	dev_dst_unit_vector->z = cos(alpha);
}
