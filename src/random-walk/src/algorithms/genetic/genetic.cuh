#pragma once

#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/validators/abstract_validator.h"

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
			virtual bool run(vector3** result, int N) override;
		protected:
			algorithms::validators::abstract_validator& validator;
			vector3* dev_unit_vectors = nullptr;
			vector3* dev_particles = nullptr;
			int* dev_generation_idx = nullptr;
			int* dev_fitness = nullptr;
			curandState* dev_states = nullptr;

			bool init(int N, int population);
			bool run(int N, int population, vector3** particles);
			void first_generation(int N, int population);
			void next_generation(int N, int population);
			void fitness_function(int N, int population);
			int select_population(int N, int population);
			void copy_solution(int N, int idx, vector3** particles);
			void terminate();
		};
	}
}