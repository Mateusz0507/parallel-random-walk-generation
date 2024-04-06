#pragma once

#include <string>
#include <sstream>

#include "common/common.h"


#define DEFAULT_PARAMS_COUNT 2

using namespace std;

namespace program_parametrization
{
	struct parameters {
	int length;
	int method;
	};
	bool read(int argc, char** argv, parameters& params);
};

