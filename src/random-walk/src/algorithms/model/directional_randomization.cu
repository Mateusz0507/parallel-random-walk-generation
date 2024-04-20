#pragma once
#include "algorithms/model/directional_randomization.cuh"


bool algorithms::directional_randomization::generate_starting_points(algorithms::model::particle* dev_points, const int N)
{
    algorithms::model::particle* starting_points = new algorithms::model::particle[N];

    curandState* dev_states = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_states, N * sizeof(curandState))))
    {
        dev_states = nullptr;
        return false;
    }

    algorithms::model::particle* dev_unit_vectors = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_unit_vectors, N * sizeof(algorithms::model::particle))))
    {
        dev_unit_vectors = nullptr;
        return false;
    }

    /* Generate starting points */

    int number_of_blocks = (N + EN_BLOCK_SIZE - 1) / EN_BLOCK_SIZE;
    algorithms::directional_randomization::kernel_setup << <number_of_blocks, EN_BLOCK_SIZE >> > (dev_states, N, std::time(nullptr), 0);
    cuda_check_terminate(cudaDeviceSynchronize());

    algorithms::directional_randomization::kernel_generate_random_unit_vectors << <number_of_blocks, EN_BLOCK_SIZE >> > (dev_unit_vectors, dev_states, N);
    algorithms::model::particle init = { 0.0, 0.0, 0.0 };

    // thrust no operator matches error resolved here https://stackoverflow.com/questions/18123407/cuda-thrust-reduction-with-double2-arrays
    // eventually thrust does not implement operator+ for float3 or double3
    thrust::device_ptr<algorithms::model::particle> dev_unit_vectors_ptr = thrust::device_ptr<algorithms::model::particle>(dev_unit_vectors);
    thrust::device_ptr<algorithms::model::particle> dev_points_ptr = thrust::device_ptr<algorithms::model::particle>(dev_points);
    algorithms::model::add_particles add;
    cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_unit_vectors_ptr, dev_unit_vectors_ptr + N, dev_points_ptr, init, add));

    if (dev_unit_vectors)
    {
        cuda_check_terminate(cudaFree(dev_unit_vectors));
        dev_unit_vectors = nullptr;
    }
    if (dev_states)
    {
        cuda_check_terminate(cudaFree(dev_states));
        dev_states = nullptr;
    }

    return true;
}

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
