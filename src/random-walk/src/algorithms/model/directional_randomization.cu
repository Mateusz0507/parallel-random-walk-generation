#pragma once
#include "algorithms/model/directional_randomization.cuh"

__global__ void algorithms::directional_randomization::kernel_setup(curandState* dev_states, int N, uint64_t seed = 0, uint64_t offset = 0)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N) curand_init(seed, tid, offset, &dev_states[tid]);
}

__global__ void algorithms::directional_randomization::kernel_generate_random_unit_vectors(algorithms::model::particle* dev_unit_vectors, curandState* dev_states, int N, int k)
{
	/*
		Article that describes uniform distribution on a sphere:
		https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
	*/

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
	{
		/* alpha and beta are in [0, pi] */
		real_t alpha = acos(2 * cuda_rand_uniform(&dev_states[tid]) - 1.0);
		real_t beta = cuda_rand_uniform(&dev_states[tid]) * PI;

		for (int i = 0; i < k; i++)
		{
			alpha = acos(M_2_PI * alpha - 1.0);
			beta = acos(M_2_PI * beta - 1.0);
		}

		/* Final beta value is in [0, 2*pi] */
		beta *= 2;

		dev_unit_vectors[tid].x = sin(alpha) * cos(beta);
		dev_unit_vectors[tid].y = sin(alpha) * sin(beta);
		dev_unit_vectors[tid].z = cos(alpha);
	}
}
