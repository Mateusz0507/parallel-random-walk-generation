#include "common/common.cuh"
#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/model/randomization.cuh"
#include "algorithms/energetic/validators/abstract_validator.h"
#include "algorithms/constaces/math_constances.h"

#include "curand_kernel.h"

#define EN_NUMERIC_EPSILON std::numeric_limits<real_t>::epsilon()
#define EN_PRECISION (100 * EN_NUMERIC_EPSILON)
#define EN_MAX_ITERATIONS 100

namespace algorithms
{
	namespace energetic
	{
		class normalisation_method : public abstract_method
		{
		private: 
			validators::abstract_validator& validator;
			model::particle* dev_unit_vectors = nullptr;
			model::particle* dev_points = nullptr;

			bool main_loop(int N, int max_iterations);
			bool allocate_memory(int N);
			void release_memory();
			bool generate_random_unit_vectors(int N);
		public:
			normalisation_method(validators::abstract_validator& validator);
			virtual bool run(model::particle** result, int N) override;
		};
	}
}