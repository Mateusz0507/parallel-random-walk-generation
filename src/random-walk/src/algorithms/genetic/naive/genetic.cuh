#pragma once

#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/validators/abstract_validator.h"
#include "algorithms/model/randomization.cuh"

#include "thrust/device_ptr.h"

#include <random>
#include <vector>

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
			genetic_method();

			virtual bool run(vector3** result, void* parameters) override;

			struct parameters {
				int N;
				int generation_size;
				float mutation_ratio = G_DEFAULT_MUTATION_RATIO;
			};

		protected:
			int N1; // number of unit vectors not particles!
			int generation_size;
			float mutation_ratio;
			int* fitness;
			int* new_generation_idx;
			int number_of_blocks;
			int generation_number_of_blocks;

			std::random_device rand_device;
			std::mt19937 generator;
			std::uniform_int_distribution<> first_parent_distribution;
			std::uniform_int_distribution<> second_parent_distribution;
			std::uniform_int_distribution<> crossover_point_distribution;

			vector3* dev_random_walk = nullptr;
			vector3* dev_chromosomes = nullptr;
			int* dev_generation_idx = nullptr;
			int* dev_fitness = nullptr;
			int* dev_invalid_distances;

			thrust::device_ptr<vector3> dev_random_walk_ptr;
			std::vector<thrust::device_ptr<vector3>> dev_chromosomes_ptrs;
			thrust::device_ptr<int> dev_generation_idx_ptr;
			thrust::device_ptr<int> dev_fitness_ptr;
			thrust::device_ptr<int> dev_invalid_distances_ptr;

			curandState* dev_states = nullptr;

			vector3 init_point = { 0.0, 0.0, 0.0 };
			model::add_vector3 add;

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