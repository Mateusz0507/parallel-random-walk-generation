#pragma once

#include <fstream>
#include <iomanip>
#include "algorithms/model/particle.cuh"

bool create_pdb_file(algorithms::model::particle* points, const int N);
