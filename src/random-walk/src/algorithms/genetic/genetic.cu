#include "algorithms/genetic/genetic.cuh"
#include "algorithms/model/randomization.cuh"
#include "thrust/scan.h"

bool algorithms::genetic::genetic_method::init(parameters* params)
{
	N = params->N;
	generation_size = params->generation_size;
	
	bool allocation_failure = false;
	fitness = new int[N];
	if (!fitness) 
	{
		allocation_failure = true;
	}
	cuda_allocate((void**)&dev_generation_idx, 2 * params->generation_size * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_fitness, 2 * params->generation_size * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_chromosomes, 2 * params->N * params->generation_size * sizeof(vector3), &allocation_failure);
	cuda_allocate((void**)&dev_states, params->N * sizeof(curandState), &allocation_failure);
	if (!allocation_failure)
	{
		algorithms::randomization::kernel_setup(dev_states, N, time(0), 0); // refactor passing arguments
	}
	return !allocation_failure;
}

bool algorithms::genetic::genetic_method::run(vector3** particles, void* params)
{
	if (init((parameters*)params))
	{
		int iteration = 0; 
		first_generation();
		fitness_function();
		int solution_idx = select_population(iteration++);
		while (solution_idx < 0) 
		{
			next_generation();
			fitness_function();
			solution_idx = select_population(iteration++);
		}
		copy_solution(particles, solution_idx);
		terminate();
		return true;
	}
	return false;
}

__global__ void kernel_init_idx(int generation_size, int* dev_generation_idx)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < generation_size)
	{
		dev_generation_idx[tid] = 0;
	}
	else if (tid < 2 * generation_size)
	{
		dev_generation_idx[tid] = tid - generation_size;
	}
}

void algorithms::genetic::genetic_method::first_generation()
{
	int number_of_blocks = (N + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE;
	for (int i = 0; i < generation_size; i++)
	{
		algorithms::randomization::kernel_generate_random_unit_vectors<<<number_of_blocks, G_BLOCK_SIZE>>>(dev_particles, dev_states, N);
		cuda_check_terminate(cudaDeviceSynchronize());
		cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_chromosomes + i * N, dev_chromosomes + (i + 1) * N, dev_chromosomes + i * N));
	}
	int generation_number_of_blocks = (generation_size + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE;
	kernel_init_idx << <generation_number_of_blocks, G_BLOCK_SIZE >> > (generation_size, dev_generation_idx);
	cuda_check_terminate(cudaDeviceSynchronize());
}

void algorithms::genetic::genetic_method::next_generation()
{
	 
}


__global__ void kernel_validate(const vector3* dev_data, int N, const real_t distance, const real_t precision, int* dev_is_invalid)
{
	const int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < N)
	{
		const int low_range = (N - 1) >> 1;
		const int high_range = N - 1 - low_range;
		const int range = index < (N >> 1) ? high_range : low_range;

		int invalid_count = 0;

		if (index + 1 != N && abs(algorithms::model::get_distance(dev_data[index], dev_data[index + 1]) - distance) > precision)
		{
			// case when the following vector3 is in different distance than the specified as an parameter
			invalid_count++;
		}

		for (int i = index + 2, j = i; i < index + range + 1; i++, j++)
		{
			if (j >= N)
			{
				j -= N;
			}
			if (algorithms::model::get_distance(dev_data[index], dev_data[j]) < distance - precision)
			{
				invalid_count++;
			}
		}

		dev_is_invalid[index] = invalid_count;
	}
}


void algorithms::genetic::genetic_method::fitness_function()
{
	for(int i = 0; i < )

}

void algorithms::genetic::genetic_method::copy_solution(vector3** particles, int idx)
{
	cuda_check_terminate(cudaMemcpy(*particles, dev_chromosomes + N * idx, N * sizeof(vector3), cudaMemcpyDeviceToHost));
}

void algorithms::genetic::genetic_method::terminate()
{
	cuda_release((void**)&dev_fitness);
	cuda_release((void**)&dev_generation_idx);
	cuda_release((void**)&dev_chromosomes);
	cuda_release((void**)&dev_states);
	delete[] fitness;
}