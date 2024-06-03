#pragma once

#include <string>
#include <sstream>

#include "common/common.h"

namespace program_parametrization
{
	struct parameters
	{
		char* method = "normalization";
		int N = 100;
		int directional_level = 0;
		int segments_number = 1;

		float mutation_ratio = 0.05;
		int generation_size = 10;
	};

	void print_usage(const char* name);
	bool read(int argc, char** argv, parameters& params);
};
