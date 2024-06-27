#include "thrust/scan.h"
#include "thrust/reduce.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "thrust/iterator/zip_iterator.h"

#include "algorithms/genetic/improved/genetic_improved.cuh"
#include "algorithms/model/particle.cuh"
#include "algorithms/genetic/kernels.cuh"
#include "chimera/chimera.h"

#include <iostream>
#include <csignal>

bool algorithms::genetic::genetic_improved_method::init(parameters* params)
{
	N1 = params->N - 1;
	generation_size = params->generation_size;
	mutation_ratio = params->mutation_ratio;
	number_of_blocks = (N1 + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE;
	generation_number_of_blocks = (2 * generation_size + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE;

	bool allocation_failure = false;

	fitness = new int[generation_size];
	if (!fitness)
	{
		allocation_failure = true;
		fitness = nullptr;
	}
	new_generation_idx = new int[generation_size];
	if (!allocation_failure && !new_generation_idx)
	{
		allocation_failure = true;
		new_generation_idx = nullptr;
	}

	cuda_allocate((void**)&dev_generation_idx, 2 * params->generation_size * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_fitness, 2 * params->generation_size * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_chromosomes, 2 * (params->N - 1) * params->generation_size * sizeof(vector3), &allocation_failure);
	cuda_allocate((void**)&dev_random_walk, params->N * sizeof(vector3), &allocation_failure);
	cuda_allocate((void**)&dev_states, (params->N - 1) * sizeof(curandState), &allocation_failure);
	cuda_allocate((void**)&dev_valid_points, params->N * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_random_walk_idx, params->N * sizeof(int), &allocation_failure);

	if (!allocation_failure)
	{
		// initializing curand
		algorithms::randomization::kernel_setup << < number_of_blocks, G_BLOCK_SIZE >> > (dev_states, N1, time(0), 0); // refactor passing arguments
		cuda_check_terminate(cudaDeviceSynchronize());

		// initializing cpu random engine
		first_parent_distribution = std::uniform_int_distribution<>(0, generation_size - 1);
		second_parent_distribution = std::uniform_int_distribution<>(0, generation_size - 2 >= 0 ? generation_size - 2 : 0);
		crossover_point_distribution = std::uniform_int_distribution<>(0, N1);

		dev_random_walk_ptr = thrust::device_ptr<vector3>(dev_random_walk);
		for (int i = 0; i < 2 * generation_size; i++)
		{
			dev_chromosomes_ptrs.push_back(thrust::device_ptr<vector3>(dev_chromosomes + i * N1));
		}
		dev_generation_idx_ptr = thrust::device_ptr<int>(dev_generation_idx);
		dev_fitness_ptr = thrust::device_ptr<int>(dev_fitness);
		dev_valid_points_ptr = thrust::device_ptr<int>(dev_valid_points);
		dev_random_walk_idx_ptr = thrust::device_ptr<int>(dev_random_walk_idx);
	}
	else
	{
		terminate();
	}

	return !allocation_failure;
}

algorithms::genetic::genetic_improved_method::genetic_improved_method()
	: rand_device{}, generator{ rand_device() }
{

}

bool algorithms::genetic::genetic_improved_method::run(vector3** particles, void* params)
{
	if (init((parameters*)params))
	{
		int iteration = 0;
		first_generation();
		int solution_idx = -1;
		while (solution_idx < 0)
		{
			next_generation();
			//print_device_array(dev_fitness, 2 * generation_size, datatype::integer);
			compute_fitness_function();
			//print_device_array(dev_fitness, 2 * generation_size, datatype::integer);
			solution_idx = select_population();
			std::cout << "Iteration " << ++iteration << std::endl;
			// print_state();
		}
		copy_solution(particles, solution_idx);
		terminate();
		return true;
	}
	return false;
}

void algorithms::genetic::genetic_improved_method::first_generation()
{
	// generating random walks
	for (int i = 0; i < generation_size; i++)
	{
		algorithms::randomization::kernel_generate_random_unit_vectors << <number_of_blocks, G_BLOCK_SIZE >> > (dev_chromosomes + i * N1, dev_states, N1);
		cuda_check_terminate(cudaDeviceSynchronize());
	}

	// initializing idx and fitness function arrays
	kernel_init_tables << <generation_number_of_blocks, G_BLOCK_SIZE >> > (generation_size, dev_generation_idx, dev_fitness);
	cuda_check_terminate(cudaDeviceSynchronize());

	// computing fitness function	
	for (int i = 0; i < generation_size; i++)
	{
		fitness_function(i, i);
	}
	cuda_check_terminate(cudaMemcpy(dev_fitness, fitness, generation_size * sizeof(int), cudaMemcpyHostToDevice));
}

// idx is an index in dev_generation_idx
__global__ void kernel_improved_crossover_and_mutate(vector3* dev_chromosomes, int N1, int child_idx, int first_parent_idx, int second_parent_idx, int* dev_fitness_function, int* dev_generation_idx,
	int generation_size, float mutation_ratio, curandState* dev_states)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < N1 && child_idx < 2 * generation_size && first_parent_idx < 2 * generation_size && second_parent_idx < 2 * generation_size)
	{
		int crossover_point = -dev_fitness_function[first_parent_idx];
		dev_chromosomes[tid + dev_generation_idx[child_idx] * N1] =
			dev_chromosomes[tid + dev_generation_idx[tid < crossover_point ? first_parent_idx : second_parent_idx] * N1];

		// mutation
		if (tid == crossover_point || curand_uniform(&dev_states[tid]) <= mutation_ratio)
		{
			int i = 0;
			algorithms::randomization::generate_random_unit_vector(&dev_chromosomes[tid + dev_generation_idx[child_idx] * N1], &dev_states[tid]);
			int idx = tid + dev_generation_idx[child_idx] * N1;

			while (tid > 0 && ++i < 15 && 
				algorithms::model::dot(dev_chromosomes[idx], dev_chromosomes[idx - 1]) < -0.5)
			{
				algorithms::randomization::generate_random_unit_vector(&dev_chromosomes[tid + dev_generation_idx[child_idx] * N1], &dev_states[tid]);
			}
		}
	}
}

void algorithms::genetic::genetic_improved_method::next_generation()
{
	for (int i = generation_size; i < 2 * generation_size; i++)
	{
		int first_parent_idx = i % ((generation_size >> 3) > 0 ? (generation_size >> 3) : 1);
		int second_parent_idx = second_parent_distribution(generator);
		if (second_parent_idx >= first_parent_idx)
		{
			second_parent_idx++;
		}
		int crossover_point = crossover_point_distribution(generator);
		kernel_improved_crossover_and_mutate << <number_of_blocks, G_BLOCK_SIZE >> > (dev_chromosomes, N1, i, first_parent_idx, second_parent_idx,
			dev_fitness, dev_generation_idx, generation_size, mutation_ratio, dev_states);
		cuda_check_terminate(cudaDeviceSynchronize());
	}
}

void algorithms::genetic::genetic_improved_method::compute_fitness_function()
{
	cuda_check_terminate(cudaMemcpy(new_generation_idx, dev_generation_idx + generation_size, generation_size * sizeof(int), cudaMemcpyDeviceToHost));
	for (int i = 0; i < generation_size; i++)
	{
		fitness_function(i, new_generation_idx[i]);
	}
	cuda_check_terminate(cudaMemcpy(dev_fitness + generation_size, fitness, generation_size * sizeof(int), cudaMemcpyHostToDevice));
}

void algorithms::genetic::genetic_improved_method::fitness_function(int fitness_idx, int chromosome_idx)
{
	cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_chromosomes_ptrs[chromosome_idx], dev_chromosomes_ptrs[chromosome_idx] + N1 + 1, dev_random_walk_ptr, init_point, add));

	kernel_improved_fitness_function << <number_of_blocks, G_BLOCK_SIZE >> > (dev_random_walk, N1 + 1, DISTANCE, G_PRECISSION, dev_valid_points, dev_random_walk_idx);
	cuda_check_terminate(cudaDeviceSynchronize());

	cuda_check_errors_status_terminate(thrust::sort_by_key(dev_valid_points_ptr, dev_valid_points_ptr + N1 + 1, dev_random_walk_idx_ptr));

	int fitness_function_value;
	cuda_check_terminate(cudaMemcpy(&fitness_function_value, dev_random_walk_idx, sizeof(int), cudaMemcpyDeviceToHost));

	fitness[fitness_idx] = fitness_function_value ? -fitness_function_value : -N1 - 1;
}

int algorithms::genetic::genetic_improved_method::select_population()
{
	cuda_check_errors_status_terminate(thrust::sort_by_key(dev_fitness_ptr, dev_fitness_ptr + 2 * generation_size, dev_generation_idx_ptr));

	int best_fitness_function;
	cuda_check_terminate(cudaMemcpy(&best_fitness_function, dev_fitness, sizeof(int), cudaMemcpyDeviceToHost));

	if (best_fitness_function > -N1)
	{
		std::cout << best_fitness_function << std::endl;
		return -1;
	}
	int best_idx;
	cuda_check_terminate(cudaMemcpy(&best_idx, dev_generation_idx, sizeof(int), cudaMemcpyDeviceToHost));
	return best_idx;
}

void algorithms::genetic::genetic_improved_method::copy_solution(vector3** particles, int idx)
{
	cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_chromosomes_ptrs[idx], dev_chromosomes_ptrs[idx] + N1 + 1, dev_random_walk, init_point, add));
	cuda_check_terminate(cudaMemcpy(*particles, dev_random_walk, (N1 + 1) * sizeof(vector3), cudaMemcpyDeviceToHost));
}

void algorithms::genetic::genetic_improved_method::terminate()
{
	cuda_release((void**)&dev_fitness);
	cuda_release((void**)&dev_generation_idx);
	cuda_release((void**)&dev_chromosomes);
	cuda_release((void**)&dev_states);
	cuda_release((void**)&dev_random_walk);
	cuda_release((void**)&dev_valid_points);
	cuda_release((void**)&dev_random_walk_idx);
	if (fitness)
	{
		delete[] fitness;
	}
	if (new_generation_idx)
	{
		delete[] new_generation_idx;
	}
}

