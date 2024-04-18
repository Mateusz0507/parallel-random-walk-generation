#pragma once

#include <fstream>
#include <iomanip>

#include "constants.h"
#include "algorithms/model/particle.cuh"

bool open_chimera(const std::string file_name);
bool create_pdb_file(algorithms::model::particle* points, const int N, const std::string file_name);
