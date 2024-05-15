#pragma once

#include <string>
#include <sstream>

#include "common/common.h"

namespace program_parametrization
{
	struct parameters
	{
		char* method = "naive";
		int N = 100;
		int directional_level = 0;
		int segments = 0;
	};

	void print_usage(const char* name);
	bool read(int argc, char** argv, parameters& params);
};
