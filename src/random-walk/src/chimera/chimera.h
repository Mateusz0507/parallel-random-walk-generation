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

#define BEFORE_PDB_FILE_NAME "before"
#define AFTER_PDB_FILE_NAME "walk"

std::string executable_path();
bool open_chimera(const std::string file_name);
bool create_pdb_file(vector3* points, const int N, const std::string file_name);
