#include "algorithms/genetic/genetic.cuh"

bool algorithms::genetic::genetic_method::init(int N, int population)
{
	bool allocation_failure = false;
	cuda_allocate((void**)&dev_generation_idx, 2 * population * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_fitness, 2 * population * sizeof(int), &allocation_failure);
	cuda_allocate((void**)&dev_particles, 2 * N * population * sizeof(vector3), &allocation_failure);
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

void algorithms::genetic::genetic_method::terminate()
{
	cuda_release((void**)&dev_fitness);
	cuda_release((void**)&dev_generation_idx);
	cuda_release((void**)&dev_particles);
}
