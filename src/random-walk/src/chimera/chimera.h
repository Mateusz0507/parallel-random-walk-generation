#pragma once

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include <fstream>
#include <iomanip>

#include "constants.h"
#include "algorithms/model/particle.cuh"

bool open_chimera(const std::string file_name);
bool create_pdb_file(vector3* points, const int N, const std::string file_name);
