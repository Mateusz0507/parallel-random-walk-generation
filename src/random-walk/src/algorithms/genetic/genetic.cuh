#pragma once

#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"

namespace algorithms 
{
	namespace genetic 
	{
		class genetic_method : public abstract_method
		{
		public:
			virtual bool run(vector3** result, int N) override;
		protected:
			vector3* dev_particles;
			int* dev_generation_idx;
			int* dev_fitness;

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