#pragma once

#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/validators/abstract_validator.h"
#include "algorithms/model/directional_randomization.cu"

#define G_PRECISSION std::numeric_limits<real_t>::epsilon()
#define G_DEFAULT_MUTATION_RATIO 0.05f

#define G_BLOCK_SIZE 64

namespace algorithms 
{
	namespace genetic 
	{
		class genetic_method : public abstract_method
		{
		public:
			virtual bool run(vector3** result, void* parameters) override;

			struct parameters {
				int N;
				int generation_size;
				float mutation_ratio = G_DEFAULT_MUTATION_RATIO;
			};

		protected:
			int N;
			int generation_size;
			float mutation_ratio;
			int* fitness;
			int* new_generation_idx;

			vector3* dev_random_walk = nullptr;
			vector3* dev_chromosomes = nullptr;
			int* dev_generation_idx = nullptr;
			int* dev_fitness = nullptr;
			int* dev_invalid_distances;
			curandState* dev_states = nullptr;

			bool init(parameters* parameters);
			void first_generation();
			void next_generation();
			void compute_fitness_function();
			void fitness_function(int fitness_idx, int chromosome_idx);
			int select_population();
			void copy_solution(vector3** particles, int idx);
			void terminate();
		};
	}
}