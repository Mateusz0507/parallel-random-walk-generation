#include "common/common.cuh"
#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/model/directional_randomization.cuh"
#include "algorithms/validators/abstract_validator.h"
#include "algorithms/constaces/math_constances.h"

#include "chimera/chimera.h"

#include "curand_kernel.h"
#include "thrust/scan.h"
#include "thrust/device_ptr.h"

#include <ctime>
#include <iostream>

#define EN_NUMERIC_EPSILON std::numeric_limits<real_t>::epsilon()
#define EN_PRECISION (100 * EN_NUMERIC_EPSILON)
#define EN_MAX_ITERATIONS 500

#define EN_BLOCK_SIZE 256

// #define _CONST_SEED
#define _TIME_SEED

#ifdef _CONST_SEED
	#define SEED 0
#elif defined(_TIME_SEED)
	#define SEED std::time(nullptr)
#endif 

#define _CONST_OFFSET

#ifdef _CONST_OFFSET
	#define OFFSET 0
#endif 

namespace algorithms
{
	namespace energetic
	{
		class normalisation_method : public abstract_method
		{
		public:
			struct parameters
			{
				int N;
				int directional_level;
				int segments_number;
			};

			normalisation_method(validators::abstract_validator& validator);
			virtual bool run(vector3** result, void*) override;
		private: 
			validators::abstract_validator& validator;
			vector3* dev_unit_vectors = nullptr;
			vector3* dev_points = nullptr;

			model::add_vector3 add;
			vector3 starting_point = { 0.0, 0.0, 0.0 };

			bool main_loop(parameters* p, int max_iterations);
			bool allocate_memory(int N);
			void release_memory();
		};
	}
}
