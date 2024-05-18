#include "algorithms/genetic/genetic.cuh"
#include "algorithms/model/randomization.cuh"
#include "thrust/scan.h"

bool algorithms::genetic::genetic_method::init(int N, int population)
{
	bool allocation_failure = false;
	cuda_allocate((void**)&dev_generation_idx, 2 * population * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_fitness, 2 * population * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_particles, 2 * N * population * sizeof(vector3), &allocation_failure);
	cuda_allocate((void**)&dev_states, N * sizeof(curandState), &allocation_failure);
	if (!allocation_failure)
	{
		algorithms::randomization::kernel_setup(dev_states, N, time(0), 0); // refactor passing arguments
	}
	return !allocation_failure;
}

bool algorithms::genetic::genetic_method::run(int N, int population, vector3** particles)
{
	if (init(N, population))
	{
		first_generation(N, population);
		fitness_function(N, population);
		int solution_idx = select_population(N, population);
		while (solution_idx < 0) {
			next_generation(N, population);
			fitness_function(N, population);
			solution_idx = select_population(N, population);
		}
		copy_solution(N, solution_idx, particles);
	}
	terminate();
	return false;
}

void algorithms::genetic::genetic_method::first_generation(int N, int population)
{
	int number_of_blocks = (N + G_BLOCK_SIZE - 1) / G_BLOCK_SIZE;
	for (int i = 0; i < population; i++)
	{
		algorithms::randomization::kernel_generate_random_unit_vectors<<<number_of_blocks, G_BLOCK_SIZE>>>(dev_particles, dev_states, N);
		cuda_check_terminate(cudaDeviceSynchronize());
		cuda_check_errors_status_terminate(thrust::exclusive_scan(dev_particles + i * N, dev_particles + (i + 1) * N, dev_particles + i * N));
	}
}

void algorithms::genetic::genetic_method::next_generation(int N, int population)
{
}

void algorithms::genetic::genetic_method::copy_solution(int N, int idx, vector3** particles)
{
	cuda_check_terminate(cudaMemcpy(*particles, dev_particles + N * idx, N * sizeof(vector3), cudaMemcpyDeviceToHost));
}

void algorithms::genetic::genetic_method::terminate()
{
	cuda_release((void**)&dev_fitness);
	cuda_release((void**)&dev_generation_idx);
	cuda_release((void**)&dev_particles);
	cuda_release((void**)&dev_states);
}
