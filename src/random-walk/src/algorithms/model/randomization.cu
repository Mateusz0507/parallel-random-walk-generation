#pragma once
#include "algorithms/model/randomization.cuh"

__global__ void algorithms::randomization::kernel_setup(curandState* dev_states, int N, uint64_t seed, uint64_t offset = 0)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N) curand_init(seed, tid, offset, &dev_states[tid]);
}

__global__ void algorithms::randomization::kernel_generate_random_unit_vectors(algorithms::model::particle * dev_unit_vectors, curandState * dev_states, int N)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
	{
		real_t alpha = cuda_rand(&dev_states[tid]) * PI;
		real_t beta = cuda_rand(&dev_states[tid]) * 2 * PI;
		dev_unit_vectors[tid].x = sin(alpha) * cos(beta);
		dev_unit_vectors[tid].y = sin(alpha) * sin(beta);
		dev_unit_vectors[tid].z = cos(alpha);
	}
}