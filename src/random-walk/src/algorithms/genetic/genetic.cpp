#include "algorithms/genetic/genetic.cuh"

bool algorithms::genetic::genetic_method::init(int N, int population)
{
	
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