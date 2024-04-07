#include "common/common.cuh"
#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/energetic/validators/abstract_validator.h"

#include "curand_kernel.h"

#define EN_NUMERIC_EPSILON std::numeric_limits<REAL_TYPE>::epsilon()
#define EN_PRECISION (100 * EN_NUMERIC_EPSILON)

namespace algorithms
{
	namespace energetic
	{
		class naive_method : public abstract_method
		{
		private:
			validators::abstract_validator& validator;
			model::particle* dev_points = nullptr;
			bool allocate_memory(int N);
			void release_memory();
			bool generate_random_starting_points(int N);
		public:
			naive_method(validators::abstract_validator& validator);
			virtual bool run(model::particle** result, int N) override;
		};
	}
}
