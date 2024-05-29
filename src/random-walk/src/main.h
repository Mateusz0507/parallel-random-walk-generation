#pragma once

// cpp
#include <cstdlib>

#include <iostream>
#include <fstream>
#include <iomanip>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

// inner files
#include "constants.h"
#include "parametrization/program_parametrization.h"
#include "algorithms/model/particle.cuh"
#include "algorithms/energetic/energetic.h"
#include "algorithms/validators/single_check_validator.cuh"
#include "algorithms/genetic/genetic.cuh"
#include "chimera/chimera.h"
#include "tests/tests.h"

using namespace std;
using namespace program_parametrization;
