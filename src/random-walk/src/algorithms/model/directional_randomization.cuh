#pragma once

#include "common/common.cuh"
#include "algorithms/model/particle.cuh"
#include "algorithms/model/spherical_coordinates.cuh"
#include "algorithms/model/matrix.cuh"
#include "algorithms/constaces/math_constances.h"

#include "curand_kernel.h"
#include "thrust/scan.h"
#include "thrust/device_ptr.h"

#define EN_BLOCK_SIZE 256

#ifdef _FLOAT
	#define cuda_rand_uniform(state) curand_uniform(state)
#elif define _DOUBLE
	#define cuda_rand_uniform(state) curand_uniform_double(state)
#endif

using namespace algorithms::model;

namespace algorithms
{
	namespace directional_randomization
	{
		bool generate_starting_points(vector3* dev_points, const int N, const int directional_parametr = 0, const int number_of_segments = 0);
		__global__ void generate_segments_directions(matrix* dev_segments_directions_matrices, curandState* dev_states, int number_of_segments, uint64_t seed);
		__global__ void kernel_setup(curandState* dev_states, int N, uint64_t seed, uint64_t offset);
		__global__ void kernel_generate_random_unit_vectors(vector3* dev_unit_vectors, curandState* dev_states, matrix* dev_segments_directions_matrices, int number_of_segments, int N, int k);
	}
}
