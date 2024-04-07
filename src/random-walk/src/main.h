#pragma once

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
#include "parametrization/program_parametrization.h"

using namespace std;
using namespace program_parametrization;