#pragma once
#include "algorithms/energetic/validators/single_check_validator.cuh"

__global__ void kernel_validate(const algorithms::model::particle* dev_data, int N, const float distance, const float precision, int* dev_is_invalid)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		const int low_range = (N - 1) >> 1;
		const int high_range = N - 1 - low_range;
		const int range = index < (N >> 1) ? high_range : low_range;

		int invalid_count = 0;

		if (abs(algorithms::model::get_distance(dev_data[index], dev_data[index + 1 == N ? 0 : index + 1]) - distance) > precision)
		{
			// case when the following particle is in different distance than the specified as an parameter
			invalid_count++;
		}

		for (int i = index + 2, j = i; i < index + range + 1; i++, j++)
		{
			if (j >= N) j = 0;
			if (algorithms::model::get_distance(dev_data[index], dev_data[j]) < distance - precision)
			{
				invalid_count++;
			}
		}

		dev_is_invalid[index] = invalid_count;
	}
}

bool algorithms::energetic::validators::single_check_validator::validate(model::particle* dev_data, int N, float distance, float precision)
{
	// checking parameters
	if (dev_data == nullptr || N < 1 || distance < 0 || precision < 0)
		return false;
	
	if (!prepare_device_memory(N))
		return false;

	int number_of_blocks = (N + SCV_BLOCK_SIZE - 1) / SCV_BLOCK_SIZE;

	// examining each distance between particles
	kernel_validate <<< number_of_blocks, SCV_BLOCK_SIZE >>> (dev_data, N, distance, precision, dev_is_invalid);
	cuda_check_terminate(cudaDeviceSynchronize());

	// counting how many distances are incorrect
	thrust::device_ptr<int> dev_is_valid_ptr = thrust::device_ptr<int>(dev_is_invalid);
	int invalid;
	cuda_check_errors_status_terminate(invalid = thrust::reduce(dev_is_valid_ptr, dev_is_valid_ptr + N));

	return invalid == 0;
}

bool algorithms::energetic::validators::single_check_validator::prepare_device_memory(int N)
{
	if (validation_array_size < N)
	{
		if (validation_array_size > 0)
		{
			cuda_check_terminate(cudaFree(dev_is_invalid));
		}
		cuda_check_terminate(cudaMalloc(&dev_is_invalid, N * sizeof(int)));
		validation_array_size = N;
	}
	return true;
}

algorithms::energetic::validators::single_check_validator::single_check_validator(int N)
{
	if (N > 0)
	{
		cuda_check_terminate(cudaMalloc(&dev_is_invalid, N * sizeof(int)));
		validation_array_size = N;
	}
}

algorithms::energetic::validators::single_check_validator::~single_check_validator()
{
	cuda_check_terminate(cudaFree(dev_is_invalid));
}

void algorithms::energetic::validators::print_test(int test_number, int result, int expected)
{
	std::cout << "Test " << test_number << ": " << "returned " << result << ", expected " << expected << std::endl;
}

void algorithms::energetic::validators::single_check_validator_test()
{
	int N = 3;
	real_t distance = 1.0;
	real_t precission = 100 * std::numeric_limits<real_t>::epsilon();
	single_check_validator validator(N);
	
	model::particle* dev_particles;
	int particle_size = sizeof(model::particle);
	cuda_check_terminate(cudaMalloc(&dev_particles, N * particle_size));

	// Test 1
	model::particle particles[] = { 0.0, 0.0, 0.0,
									0.0, 1.0, 0.0,
									1.0, 0.0, 0.0 };
	
	cuda_check_terminate(cudaMemcpy(dev_particles, &particles, N * particle_size, cudaMemcpyHostToDevice));

	int result = validator.validate(dev_particles, N, distance, precission);

	print_test(1, result, 1);

	// releasing memory
	cuda_check_terminate(cudaFree(dev_particles));

	
}