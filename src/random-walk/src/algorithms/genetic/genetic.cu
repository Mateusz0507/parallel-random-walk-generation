#include "algorithms/genetic/genetic.cuh"
#include "algorithms/model/randomization.cuh"
#include "thrust/scan.h"
#include "algorithms/model/particle.cuh"
#include "thrust/reduce.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "thrust/iterator/zip_iterator.h"

bool algorithms::genetic::genetic_method::init(parameters* params)
{
	N = params->N;
	generation_size = params->generation_size;
	mutation_ratio = params->mutation_ratio;
	
	bool allocation_failure = false;

	fitness = new int[generation_size];
	if (!fitness) 
	{
		allocation_failure = true;
		fitness = nullptr;
	}
	if(!allocation_failure && !new_generation_idx)
	{
		allocation_failure = true;
		new_generation_idx = nullptr;
	}

	cuda_allocate((void**)&dev_generation_idx, 2 * params->generation_size * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_fitness, 2 * params->generation_size * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_chromosomes, 2 * params->N * params->generation_size * sizeof(vector3), &allocation_failure);
	cuda_allocate((void**)&dev_random_walk, N * sizeof(vector3), &allocation_failure);
	cuda_allocate((void**)&dev_states, params->N * sizeof(curandState), &allocation_failure);

	if (!allocation_failure)
	{
		algorithms::randomization::kernel_setup(dev_states, N, time(0), 0); // refactor passing arguments
	}

	if (allocation_failure)
	{
		terminate();
	}

	return !allocation_failure;
}

bool algorithms::genetic::genetic_method::run(vector3** particles, void* params)
{
	if (init((parameters*)params))
	{
		first_generation();
		int solution_idx = -1;
		while (solution_idx < 0) 
		{
			next_generation();
			compute_fitness_function();
			solution_idx = select_population();
		}
		copy_solution(particles, solution_idx);
		terminate();
		return true;
	}
	return false;
}

__global__ void kernel_init_tables(int generation_size, int* dev_generation_idx, int* dev_fitness_function)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < 2 * generation_size)
	{
		dev_generation_idx[tid] = tid;
		dev_fitness_function[tid] = INT_MAX;
	}
}

void algorithms::genetic::genetic_method::first_generation()
{
	// generating random walks
	int number_of_blocks = (N + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE;
	for (int i = 0; i < generation_size; i++)
	{
		algorithms::randomization::kernel_generate_random_unit_vectors<<<number_of_blocks, G_BLOCK_SIZE>>>(dev_unit_vectors + i * N, dev_states, N);
		cuda_check_terminate(cudaDeviceSynchronize());
	}

	// initializing idx and fitness function arrays
	int generation_number_of_blocks = (generation_size + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE;
	kernel_init_tables << <generation_number_of_blocks, G_BLOCK_SIZE >> > (generation_size, dev_generation_idx, dev_fitness_function);
	cuda_check_terminate(cudaDeviceSynchronize());

	// computing fitness function	
	for (int i = 0; i < generation_size; i++)
	{	
		fitness_function(i, i);
	}
}

// idx is an index in dev_generation_idx
__global__ void kernel_crossover_and_mutate(vector3* dev_chromosomes, int N, int child_idx, int parent_idx1, int parent_idx2, int crossover_point, int* dev_generation_idx,
	int generation_size, float mutation_ratio, curandState* dev_states)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < N) 
	{
		dev_chromosomes[dev_generation_idx[child_idx]] =
			dev_chromosomes[dev_generation_idx[tid < crossover_point ? parent_idx1 : parent_idx2]];

		if (curand_uniform(&dev_states[tid]) <= mutation_ratio)
		{

		}

	}
}

void algorithms::genetic::genetic_method::next_generation()
{
	


}

void algorithms::genetic::genetic_method::compute_fitness_function()
{
	cuda_check_terminate(cudaMemcpy(new_generation_idx, dev_generation_idx + generation_size, generation_size * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < generation_size; i++)
	{
		fitness_function(i, new_generation_idx[i]);
	}
}

void algorithms::genetic::genetic_method::fitness_function(int fitness_idx, int chromosome_idx)
{
	thrust::exclusive_scan(dev_chromosomes + chromosome_idx * N, dev_chromosomes + (chromosome_idx + 1) * N, dev_random_walk);
	kernel_fitness_function << <number_of_blocks, G_BLOCK_SIZE >> > (dev_random_walk, N, DISTANCE, G_PRECISSION, dev_invalid_distances);
	cuda_check_terminate(cudaDeviceSynchronize());
	cuda_check_errors_status_terminate(fitness[fitness_idx] = thrust::reduce(dev_invalid_distances, dev_invalid_distances + N));
}

int algorithms::genetic::genetic_method::select_population()
{
 	cuda_check_errors_status_terminate(thrust::sort_by_key(dev_fitness, dev_fitness + N, thrust::make_zip_iterator(thrust::make_tuple(dev_fitness, dev_generation_idx))));

	int best_fitness_function;
	cudaMemcpy(&best_fitness_function, dev_fitness, sizeof(int), cudaMemcpyDeviceToHost);
	if (best_fitness_function > 0)
	{
		return -1;
	}
	int best_idx;
	cudaMemcpy(&best_idx, dev_generation_idx, sizeof(int), cudaMemcpyDeviceToHost);
	return best_idx;
}

// TODO: abstract the kernel due to its dual usage
__global__ void kernel_fitness_function(const vector3* dev_data, int N, const real_t distance, const real_t precision, int* dev_invalid_distances)
{
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < N)
	{
		const int low_range = (N - 1) >> 1;
		const int high_range = N - 1 - low_range;
		const int range = tid < (N >> 1) ? high_range : low_range;

		int invalid_count = 0;

		if (tid + 1 != N && abs(algorithms::model::get_distance(dev_data[tid], dev_data[tid + 1]) - distance) > precision)
		{
			// case when the following vector3 is in different distance than the specified as an parameter
			invalid_count++;
		}

		for (int i = tid + 2, j = i; i < tid + range + 1; i++, j++)
		{
			if (j >= N)
			{
				j -= N;
			}
			if (algorithms::model::get_distance(dev_data[tid], dev_data[j]) < distance - precision)
			{
				invalid_count++;
			}
		}

		dev_invalid_distances[tid] = invalid_count;
	}
}

void algorithms::genetic::genetic_method::copy_solution(vector3** particles, int idx)
{
	cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_chromosomes + idx * N, dev_chromosomes + (idx + 1) * N, dev_random_walk));
	cuda_check_terminate(cudaMemcpy(*particles, dev_random_walk, N * sizeof(vector3), cudaMemcpyDeviceToHost));
}

void algorithms::genetic::genetic_method::terminate()
{
	cuda_release((void**)&dev_fitness);
	cuda_release((void**)&dev_generation_idx);
	cuda_release((void**)&dev_chromosomes);
	cuda_release((void**)&dev_states);
	cuda_release((void**)&dev_random_walk);
	cuda_release((void**)&dev_invalid_distances);
	if (fitness)
	{
		delete[] fitness;
	}
	if (new_generation_idx)
	{
		delete[] new_generation_idx;
	}
}
