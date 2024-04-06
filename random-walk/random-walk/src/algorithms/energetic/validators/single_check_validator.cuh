#pragma once

// #include "global/global.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/energetic/validators/abstract_validator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SCV_BLOCK_SIZE 256

namespace algorithms
{
	namespace energetic
	{
		namespace validators
		{
			class single_check_validator: public abstract_validator
			{
			private:
				bool* dev_is_valid;
			public:
				single_check_validator();
				~single_check_validator();
				virtual bool validate(model::particle* dev_data, int N, float distance, float precision) override;
			};

		}
	}
}

