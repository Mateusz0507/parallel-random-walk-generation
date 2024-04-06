#pragma once
#include "algorithms/energetic/validators/single_check_validator.cuh"

__global__ void kernel_validate(algorithms::model::particle* dev_data, int N, float distance, float precision, bool* dev_is_valid)
{
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		const int low_range = (N - 1) >> 1;
		const int high_range = N - 1 - low_range;
		const int range = index < (N >> 1) ? high_range : low_range;

		if (abs(algorithms::model::get_distance(dev_data[index], dev_data[index + 1]) - distance) > precision)
		{
			// case when the following particle is in different distance than the specified as an parameter
			*dev_is_valid = false;
		}

		for (int i = index + 2; i < index + range + 1; i++)
		{
			if (algorithms::model::get_distance(dev_data[index], dev_data[i]) < distance - precision)
			{
				// TODO: some smart passing information about the points that are too close to each other
			}
		}
	}
}

bool algorithms::energetic::validators::single_check_validator::validate(model::particle* dev_data, int N, float distance, float precision)
{
	int number_of_blocks = (N + SCV_BLOCK_SIZE - 1) / SCV_BLOCK_SIZE;
	bool tmp = true;
	cudaMemcpy(dev_is_valid, &tmp, sizeof(tmp), cudaMemcpyHostToDevice);
	// TODO: Error handling

	kernel_validate <<< number_of_blocks, SCV_BLOCK_SIZE >>> (dev_data, N, distance, precision, dev_is_valid);
	cudaDeviceSynchronize();
	// TODO: Error handling

	cudaMemcpy(&tmp, dev_is_valid, sizeof(tmp), cudaMemcpyDeviceToHost);
	// TODO: Error handling

	return tmp;
}

algorithms::energetic::validators::single_check_validator::single_check_validator()
{
	cudaMalloc(&dev_is_valid, sizeof(bool));
	// TODO: Error handling
}

algorithms::energetic::validators::single_check_validator::~single_check_validator()
{
	cudaFree(dev_is_valid);
}