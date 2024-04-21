#include "common/common.cuh"
#include "algorithms/abstract_method.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/model/directional_randomization.cuh"
#include "algorithms/energetic/validators/abstract_validator.h"
#include "chimera/chimera.h";

#include "curand_kernel.h"

#define EN_NUMERIC_EPSILON std::numeric_limits<real_t>::epsilon()
#define EN_PRECISION (100 * EN_NUMERIC_EPSILON)
#define EN_BLOCK_SIZE 256

namespace algorithms
{
	namespace energetic
	{
		class naive_method : public abstract_method
		{
		private:
			validators::abstract_validator& validator;
			vector3* dev_points = nullptr;
			bool allocate_memory(int N);
			void release_memory();
		public:
			naive_method(validators::abstract_validator& validator);
			virtual bool run(vector3** result, int N) override;
		};
	}
}
