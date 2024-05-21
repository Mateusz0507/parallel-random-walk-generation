#pragma once

#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/validators/abstract_validator.h"

#define G_PRECISSION std::numeric_limits<real_t>::epsilon()

#define G_BLOCK_SIZE 64

namespace algorithms 
{
	namespace genetic 
	{
		class genetic_method_parameters {
			int N;
			int population;
		};

		class genetic_method : public abstract_method
		{
		public:
			virtual bool run(vector3** result, void* parameters) override;

			struct parameters {
				int N;
				int generation_size;
			};
		protected:
			int N;
			int generation_size;
			int* fitness;
			int* new_generation_idx; 

			vector3* dev_unit_vectors = nullptr;
			vector3* dev_chromosomes = nullptr;
			int* dev_generation_idx = nullptr;
			int* dev_fitness = nullptr;
			int* dev_invalid_distances;
			curandState* dev_states = nullptr;

			bool init(parameters* parameters);
			void first_generation();
			void next_generation();
			void fitness_function();
			int select_population(int iteration);
			void copy_solution(vector3** particles, int idx);
			void terminate();
		};
	}
}