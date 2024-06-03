#pragma once
#include "algorithms/model/directional_randomization.cuh"


bool algorithms::directional_randomization::generate_starting_positions(
    vector3* dev_unit_vectors_argument,
    vector3* dev_points_argument,
    const int N,
    const int directional_parametr,
    const int number_of_segments,
    const int seed)
{
    /* Case when number of segments is too small */
    if (number_of_segments < 1)
        return false;

    /* Case when number of segments is too big */
    if (directional_parametr > 0 && N - 1 < number_of_segments)
        return false;

    int number_of_blocks = (N + EN_BLOCK_SIZE - 1) / EN_BLOCK_SIZE;


    curandState* dev_states = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_states, (N - 1) * sizeof(curandState))))
    {
        dev_states = nullptr;
        return false;
    }

    algorithms::directional_randomization::kernel_setup <<<number_of_blocks, EN_BLOCK_SIZE>>>(dev_states, N - 1, seed);
    cuda_check_terminate(cudaDeviceSynchronize());

    vector3* dev_points = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_points, N * sizeof(vector3))))
    {
        dev_points = nullptr;
        return false;
    }

    vector3* dev_unit_vectors = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_unit_vectors, (N - 1) * sizeof(vector3))))
    {
        dev_unit_vectors = nullptr;
        return false;
    }

    matrix* dev_segments_directions_matrices = nullptr;
    if (!cuda_check_continue(cudaMalloc(&dev_segments_directions_matrices, number_of_segments * sizeof(matrix))))
    {
        dev_segments_directions_matrices = nullptr;
        return false;
    }


    kernel_generate_segments_directions <<<number_of_blocks, EN_BLOCK_SIZE>>>
        (dev_segments_directions_matrices, dev_states, number_of_segments, seed);
    
    kernel_generate_random_unit_vectors <<<number_of_blocks, EN_BLOCK_SIZE>>>
        (dev_unit_vectors, dev_states, dev_segments_directions_matrices, number_of_segments, N - 1, directional_parametr);


    /*
    * thrust no operator matches error resolved here:
    * https://stackoverflow.com/questions/18123407/cuda-thrust-reduction-with-double2-arrays
    * eventually thrust does not implement operator+ for float3 or double3
    */
    thrust::device_ptr<vector3> dev_unit_vectors_ptr = thrust::device_ptr<vector3>(dev_unit_vectors);
    thrust::device_ptr<vector3> dev_points_ptr = thrust::device_ptr<vector3>(dev_points);
    add_vector3 add;
    cuda_check_errors_status_terminate(thrust::inclusive_scan(dev_unit_vectors_ptr, dev_unit_vectors_ptr + N, dev_points_ptr + 1, add));


    cudaMemcpy(dev_unit_vectors_argument, dev_unit_vectors, (N - 1) * sizeof(vector3), cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_points_argument, dev_points, N * sizeof(vector3), cudaMemcpyDeviceToDevice);


    if (dev_states)
    {
        cuda_check_terminate(cudaFree(dev_states));
        dev_states = nullptr;
    }

    if (dev_points)
    {
        cuda_check_terminate(cudaFree(dev_points));
        dev_points = nullptr;
    }

    if (dev_unit_vectors)
    {
        cuda_check_terminate(cudaFree(dev_unit_vectors));
        dev_unit_vectors = nullptr;
    }

    if (dev_segments_directions_matrices)
    {
        cuda_check_terminate(cudaFree(dev_segments_directions_matrices));
        dev_segments_directions_matrices = nullptr;
    }

    return true;
}

__global__ void algorithms::directional_randomization::kernel_setup(
    curandState* dev_states, int n, uint64_t seed, uint64_t offset)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n)
        curand_init(seed, index, offset, &dev_states[index]);
}

__global__ void algorithms::directional_randomization::kernel_generate_segments_directions(
    matrix* dev_segments_directions_matrices, curandState* dev_states, int number_of_segments, uint64_t seed)
{
    /*
    * Generation of starting points can be directed towards [1, 0, 0] direction.
    * To change direction towards vector v, you need to change basis of linear space so v is [1, 0, 0] in new basis.
    */
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < number_of_segments)
    {
        real_t alpha = acos(2 * cuda_rand_uniform(&dev_states[tid]) - 1.0);
        real_t beta = cuda_rand_uniform(&dev_states[tid]) * 2 * PI;

        dev_segments_directions_matrices[tid] = matrix(spherical_coordinates(alpha, beta));
    }
}

__global__ void algorithms::directional_randomization::kernel_generate_random_unit_vectors(
    vector3* dev_unit_vectors,
    curandState* dev_states,
    matrix* dev_segments_directions_matrices,
    int number_of_segments,
    int n,
    int k)
{
	/*
    * Article that describes uniform distribution on a sphere:
    * https://www.bogotobogo.com/Algorithms/uniform_distribution_sphere.php
	*/

	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < n)
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

        vector3 v;
        v.x = sin(alpha) * cos(beta);
        v.y = sin(alpha) * sin(beta);
        v.z = cos(alpha);

        int index_of_segment = (tid * number_of_segments) / n;
        dev_unit_vectors[tid] = dev_segments_directions_matrices[index_of_segment].multiply(v);
	}
}
