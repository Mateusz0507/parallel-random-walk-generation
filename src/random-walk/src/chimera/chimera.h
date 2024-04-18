#pragma once

#include <fstream>
#include <iomanip>
#include "algorithms/model/particle.cuh"

bool create_file(algorithms::model::particle* points, const int N);
