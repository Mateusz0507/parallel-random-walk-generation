#pragma once

#include "common/common.cuh"
#include "algorithms/model/particle.cuh"
#include "algorithms/constaces/math_constances.h"

#include "curand_kernel.h"

#ifdef _FLOAT
	#define cuda_rand_uniform(state) curand_uniform(state)
#elif define _DOUBLE
	#define cuda_rand_uniform(state) curand_uniform_double(state)
#endif

namespace algorithms
{
	namespace randomization
	{
		// article that describes uniform distribution on a sphere https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
		__global__ void kernel_setup(curandState* dev_states, int N, uint64_t seed, uint64_t offset);
		__global__ void kernel_generate_random_unit_vectors(algorithms::model::particle* dev_unit_vectors, curandState* dev_states, int N);
	}
}
