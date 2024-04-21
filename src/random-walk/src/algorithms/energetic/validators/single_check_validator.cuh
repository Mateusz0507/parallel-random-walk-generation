#pragma once

#include "common/common.cuh"
#include "algorithms/model/particle.cuh"
#include "algorithms/energetic/validators/abstract_validator.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/device_vector.h>

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
				int* dev_is_invalid = nullptr;
				int validation_array_size = 0;

				bool prepare_device_memory(int N);
			public:
				single_check_validator(int N = 0);
				~single_check_validator();
				virtual bool validate(vector3* dev_data, int N, float distance, float precision) override;
			};

			void print_test(int test_number, int result, int expected);
			void single_check_validator_test();
		}
	}
}

